#include "common/ducklake_util.hpp"
#include "storage/ducklake_scan.hpp"
#include "storage/ducklake_multi_file_list.hpp"
#include "storage/ducklake_multi_file_reader.hpp"

#include "duckdb/common/local_file_system.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/optimizer/filter_combiner.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "storage/ducklake_table_entry.hpp"

namespace duckdb {

DuckLakeMultiFileList::DuckLakeMultiFileList(DuckLakeFunctionInfo &read_info,
                                             vector<DuckLakeDataFile> transaction_local_files_p,
                                             shared_ptr<DuckLakeInlinedData> transaction_local_data_p, string filter_p)
    : MultiFileList(vector<OpenFileInfo> {}, FileGlobOptions::ALLOW_EMPTY), read_info(read_info), read_file_list(false),
      transaction_local_files(std::move(transaction_local_files_p)),
      transaction_local_data(std::move(transaction_local_data_p)), filter(std::move(filter_p)) {
}

DuckLakeMultiFileList::DuckLakeMultiFileList(DuckLakeFunctionInfo &read_info,
                                             vector<DuckLakeFileListEntry> files_to_scan)
    : MultiFileList(vector<OpenFileInfo> {}, FileGlobOptions::ALLOW_EMPTY), read_info(read_info),
      files(std::move(files_to_scan)), read_file_list(true) {
}

DuckLakeMultiFileList::DuckLakeMultiFileList(DuckLakeFunctionInfo &read_info,
                                             const DuckLakeInlinedTableInfo &inlined_table)
    : MultiFileList(vector<OpenFileInfo> {}, FileGlobOptions::ALLOW_EMPTY), read_info(read_info), read_file_list(true) {
	DuckLakeFileListEntry file_entry;
	file_entry.file.path = inlined_table.table_name;
	file_entry.row_id_start = 0;
	file_entry.data_type = DuckLakeDataType::INLINED_DATA;
	files.push_back(std::move(file_entry));
	inlined_data_tables.push_back(inlined_table);
}

bool ValueIsFinite(const Value &val) {
	if (val.type().id() != LogicalTypeId::FLOAT && val.type().id() != LogicalTypeId::DOUBLE) {
		return true;
	}
	double constant_val = val.GetValue<double>();
	return Value::IsFinite(constant_val);
}

string CastValueToTarget(const Value &val, const LogicalType &type) {
	if (type.IsNumeric() && ValueIsFinite(val)) {
		// for (finite) numerics we directly emit the number
		return val.ToString();
	}
	// convert to a string
	return DuckLakeUtil::SQLLiteralToString(val.ToString());
}

string CastStatsToTarget(const string &stats, const LogicalType &type) {
	// we only need to cast numerics
	if (type.IsNumeric()) {
		return "TRY_CAST(" + stats + " AS " + type.ToString() + ")";
	}
	return stats;
}

string GenerateConstantFilter(const ConstantFilter &constant_filter, const LogicalType &type,
                              unordered_set<string> &referenced_stats) {
	auto constant_str = CastValueToTarget(constant_filter.constant, type);
	auto min_value = CastStatsToTarget("min_value", type);
	auto max_value = CastStatsToTarget("max_value", type);
	switch (constant_filter.comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		// x = constant
		// this can only be true if "constant BETWEEN min AND max"
		referenced_stats.insert("min_value");
		referenced_stats.insert("max_value");
		return StringUtil::Format("%s BETWEEN %s AND %s", constant_str, min_value, max_value);
	case ExpressionType::COMPARE_NOTEQUAL:
		// x <> constant
		// this can only be false if "constant = min AND constant = max" (i.e. min = max = constant)
		// skip this for now
		return string();
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		// x >= constant
		// this can only be true if "max >= C"
		referenced_stats.insert("max_value");
		return StringUtil::Format("%s >= %s", max_value, constant_str);
	case ExpressionType::COMPARE_GREATERTHAN:
		// x > constant
		// this can only be true if "max > C"
		referenced_stats.insert("max_value");
		return StringUtil::Format("%s > %s", max_value, constant_str);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		// x <= constant
		// this can only be true if "min <= C"
		referenced_stats.insert("min_value");
		return StringUtil::Format("%s <= %s", min_value, constant_str);
	case ExpressionType::COMPARE_LESSTHAN:
		// x < constant
		// this can only be true if "min < C"
		referenced_stats.insert("min_value");
		return StringUtil::Format("%s < %s", min_value, constant_str);
	default:
		// unsupported
		return string();
	}
}

string GenerateConstantFilterDouble(const ConstantFilter &constant_filter, const LogicalType &type,
                                    unordered_set<string> &referenced_stats) {
	double constant_val = constant_filter.constant.GetValue<double>();
	bool constant_is_nan = Value::IsNan(constant_val);
	switch (constant_filter.comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		// x = constant
		if (constant_is_nan) {
			// x = NAN - check for `contains_nan`
			referenced_stats.insert("contains_nan");
			return "contains_nan";
		}
		// else check as if this is a numeric
		return GenerateConstantFilter(constant_filter, type, referenced_stats);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHAN: {
		if (constant_is_nan) {
			// skip these filters if the constant is nan
			// note that > and >= we can actually handle since nan is the biggest value
			// (>= is equal to =, > is always false)
			return string();
		}
		// generate the numeric filter
		string filter = GenerateConstantFilter(constant_filter, type, referenced_stats);
		if (filter.empty()) {
			return string();
		}
		// since NaN is bigger than anything - we also need to check for contains_nan
		referenced_stats.insert("contains_nan");
		return filter + " OR contains_nan";
	}
	case ExpressionType::COMPARE_NOTEQUAL:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_LESSTHAN:
		if (constant_is_nan) {
			// skip these filters if the constant is nan
			return string();
		}
		// these are equivalent to the numeric filter
		return GenerateConstantFilter(constant_filter, type, referenced_stats);
	default:
		// unsupported
		return string();
	}
}

string GenerateFilterPushdown(const TableFilter &filter, unordered_set<string> &referenced_stats) {
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		auto &type = constant_filter.constant.type();
		switch (type.id()) {
		case LogicalTypeId::BLOB:
			return string();
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::DOUBLE:
			return GenerateConstantFilterDouble(constant_filter, type, referenced_stats);
		default:
			return GenerateConstantFilter(constant_filter, type, referenced_stats);
		}
	}
	case TableFilterType::IS_NULL:
		// IS NULL can only be true if the file has any NULL values
		referenced_stats.insert("null_count");
		return "null_count > 0";
	case TableFilterType::IS_NOT_NULL:
		// IS NOT NULL can only be true if the file has any valid values
		referenced_stats.insert("value_count");
		return "value_count > 0";
	case TableFilterType::CONJUNCTION_OR: {
		auto &conjunction_or_filter = filter.Cast<ConjunctionOrFilter>();
		string result;
		for (auto &child_filter : conjunction_or_filter.child_filters) {
			if (!result.empty()) {
				result += " OR ";
			}
			string child_str = GenerateFilterPushdown(*child_filter, referenced_stats);
			if (child_str.empty()) {
				return string();
			}
			result += child_str;
		}
		return result;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction_and_filter = filter.Cast<ConjunctionAndFilter>();
		string result;
		for (auto &child_filter : conjunction_and_filter.child_filters) {
			if (!result.empty()) {
				result += " AND ";
			}
			string child_str = GenerateFilterPushdown(*child_filter, referenced_stats);
			if (child_str.empty()) {
				return string();
			}
			result += child_str;
		}
		return result;
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &optional_filter = filter.Cast<OptionalFilter>();
		return GenerateFilterPushdown(*optional_filter.child_filter, referenced_stats);
	}
	case TableFilterType::IN_FILTER: {
		auto &in_filter = filter.Cast<InFilter>();
		string result;
		for (auto &value : in_filter.values) {
			if (!result.empty()) {
				result += " OR ";
			}
			auto temporary_constant_filter = ConstantFilter(ExpressionType::COMPARE_EQUAL, value);
			auto next_filter = GenerateFilterPushdown(temporary_constant_filter, referenced_stats);
			if (next_filter.empty()) {
				return string();
			}
			result += next_filter;
		}
		return result;
	}
	default:
		// unsupported filter
		return string();
	}
}

string DuckLakeMultiFileList::GenerateSimpleDynamicFilterPushDownQuery(const vector<column_t> &column_ids,
                                                                       TableFilterSet &filters) const {
	string filter;
	for (auto &entry : filters.filters) {
		auto column_id = entry.first;
		if (IsVirtualColumn(column_ids[column_id])) {
			// skip pushing filters on virtual columns
			continue;
		}
		// FIXME: handle structs
		auto column_index = PhysicalIndex(column_ids[column_id]);
		auto &root_id = read_info.table.GetFieldId(column_index);
		unordered_set<string> referenced_stats;
		auto new_filter = GenerateFilterPushdown(*entry.second, referenced_stats);
		if (new_filter.empty()) {
			// failed to generate filter for this column
			continue;
		}
		// generate the final filter for this column
		string final_filter;
		final_filter = "table_id=" + to_string(read_info.table_id.index);
		final_filter += " AND ";
		final_filter += "column_id=" + to_string(root_id.GetFieldIndex().index);
		final_filter += " AND ";
		final_filter += "(";
		// if any of the referenced stats are NULL we cannot prune
		for (auto &stats_name : referenced_stats) {
			final_filter += stats_name + " IS NULL OR ";
		}
		// finally add the filter
		final_filter += "(" + new_filter + "))";
		// add the filter to the list of filters
		if (!filter.empty()) {
			filter += " AND ";
		}
		filter += StringUtil::Format(
		    "data_file_id IN (SELECT data_file_id FROM {METADATA_CATALOG}.ducklake_file_column_stats WHERE %s)",
		    final_filter);
	}
	return filter;
}

// Forward declarations for complex expression filter generation
string GenerateComplexExpressionFilter(const vector<unique_ptr<Expression>> &expressions,
                                       const vector<column_t> &column_ids, const vector<idx_t> &field_ids);
string GenerateExpressionCondition(const Expression &expr, const vector<column_t> &column_ids,
                                   const vector<idx_t> &field_ids);
string GenerateConjunctionCondition(const Expression &expr, const vector<column_t> &column_ids,
                                    const vector<idx_t> &field_ids, const string &op);
string GenerateComparisonCondition(const Expression &expr, const vector<column_t> &column_ids,
                                   const vector<idx_t> &field_ids);

unique_ptr<MultiFileList> DuckLakeMultiFileList::ComplexFilterPushdown(ClientContext &context,
                                                                       const MultiFileOptions &options,
                                                                       MultiFilePushdownInfo &info,
                                                                       vector<unique_ptr<Expression>> &filters) {
	if (filters.empty()) {
		return nullptr;
	}
	if (read_info.scan_type != DuckLakeScanType::SCAN_TABLE) {
		// filter pushdown is only supported when scanning full tables
		return nullptr;
	}

	// Use FilterCombiner first to get standard table filters
	FilterCombiner combiner(context);
	for (auto riter = filters.rbegin(); riter != filters.rend(); ++riter) {
		combiner.AddFilter(riter->get()->Copy());
	}

	// Change the column_indexes to actually index into info.column_ids
	vector<ColumnIndex> modified_column_indexes = info.column_indexes;
	for (auto &col_idx : modified_column_indexes) {
		bool found = false;
		for (idx_t i = 0; i < info.column_ids.size(); ++i) {
			auto &col_id = info.column_ids[i];
			if (col_id == col_idx.GetPrimaryIndex()) {
				col_idx = ColumnIndex(i);
				found = true;
				break;
			}
		}
		if (!found) {
			throw InternalException("Failed to find column index for filter pushdown");
		}
	}

	vector<FilterPushdownResult> pushdown_results;
	auto filter_set = combiner.GenerateTableScanFilters(modified_column_indexes, pushdown_results);

	// Apply standard DynamicFilterPushdown first if we have filters
	string dynamic_filter_query = GenerateSimpleDynamicFilterPushDownQuery(info.column_ids, filter_set);

	// Check if any filters were not fully pushed down
	vector<unique_ptr<Expression>> unpushed_filters;
	for (size_t i = 0; i < filters.size() && i < pushdown_results.size(); i++) {
		if (pushdown_results[i] != FilterPushdownResult::PUSHED_DOWN_FULLY) {
			unpushed_filters.push_back(filters[i]->Copy());
		}
	}

	// If no unpushed filters, return the dynamic filter result (if any)
	if (unpushed_filters.empty()) {
		if (!dynamic_filter_query.empty()) {
			return make_uniq<DuckLakeMultiFileList>(read_info, transaction_local_files, transaction_local_data,
			                                        std::move(dynamic_filter_query));
		} else {
			return nullptr;
		}
	}

	// Try to push down complex expressions for unpushed filters
	// Build mapping from projected column index to table field id
	vector<idx_t> field_ids;
	field_ids.reserve(info.column_ids.size());
	for (auto col_id : info.column_ids) {
		if (IsVirtualColumn(col_id)) {
			// skip virtual columns
			field_ids.push_back(DConstants::INVALID_INDEX);
			continue;
		}
		auto phys_index = PhysicalIndex(col_id);
		auto &root_id = read_info.table.GetFieldId(phys_index);
		field_ids.push_back(root_id.GetFieldIndex().index);
	}
	string complex_filter = GenerateComplexExpressionFilter(unpushed_filters, info.column_ids, field_ids);

	if (complex_filter.empty()) {
		// No additional complex pushdown possible, return dynamic result
		if (!dynamic_filter_query.empty()) {
			return make_uniq<DuckLakeMultiFileList>(read_info, transaction_local_files, transaction_local_data,
			                                        std::move(dynamic_filter_query));
		} else {
			return nullptr;
		}
	}

	// Combine with existing dynamic filter if we have one
	if (!dynamic_filter_query.empty()) {
		if (!dynamic_filter_query.empty()) {
			complex_filter = dynamic_filter_query + " AND (" + complex_filter + ")";
		}
		return make_uniq<DuckLakeMultiFileList>(read_info, transaction_local_files, transaction_local_data,
		                                        std::move(complex_filter));
	} else {
		return make_uniq<DuckLakeMultiFileList>(read_info, transaction_local_files, transaction_local_data,
		                                        std::move(complex_filter));
	}
}

string GenerateComplexExpressionFilter(const vector<unique_ptr<Expression>> &expressions,
                                       const vector<column_t> &column_ids, const vector<idx_t> &field_ids) {
	if (expressions.empty()) {
		return string();
	}

	// Generate conditions for each expression
	vector<string> expression_conditions;
	for (auto &expr : expressions) {
		string expr_condition = GenerateExpressionCondition(*expr, column_ids, field_ids);
		if (!expr_condition.empty()) {
			expression_conditions.push_back(expr_condition);
		}
	}

	if (expression_conditions.empty()) {
		return string();
	}

	// Combine all expression conditions with AND
	string combined_condition = StringUtil::Join(expression_conditions, " AND ");

	// Generate the final filter query using a simple SELECT from a base query
	// This creates a filter that selects file IDs where the complex conditions are satisfied
	return StringUtil::Format("data_file_id IN (SELECT DISTINCT data_file_id FROM "
	                          "{METADATA_CATALOG}.ducklake_file_column_stats base WHERE %s)",
	                          combined_condition);
}

string GenerateExpressionCondition(const Expression &expr, const vector<column_t> &column_ids,
                                   const vector<idx_t> &field_ids) {
	switch (expr.GetExpressionType()) {
	case ExpressionType::CONJUNCTION_AND:
		return GenerateConjunctionCondition(expr, column_ids, field_ids, " AND ");
	case ExpressionType::CONJUNCTION_OR:
		return GenerateConjunctionCondition(expr, column_ids, field_ids, " OR ");
	case ExpressionType::COMPARE_EQUAL:
	case ExpressionType::COMPARE_NOTEQUAL:
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return GenerateComparisonCondition(expr, column_ids, field_ids);
	default:
		// Unsupported expression type - return empty to skip
		return string();
	}
}

string GenerateConjunctionCondition(const Expression &expr, const vector<column_t> &column_ids,
                                    const vector<idx_t> &field_ids, const string &op) {
	// Check if this is a conjunction expression
	if (expr.expression_class != ExpressionClass::BOUND_CONJUNCTION) {
		return string(); // Can't parse non-conjunction as conjunction
	}

	const auto &conj_expr = expr.Cast<BoundConjunctionExpression>();

	// Process all child expressions
	vector<string> child_conditions;
	for (const auto &child : conj_expr.children) {
		string child_condition = GenerateExpressionCondition(*child, column_ids, field_ids);
		if (!child_condition.empty()) {
			child_conditions.push_back(child_condition);
		}
	}

	// For AND: if some children can't be parsed, replace with TRUE (as requested)
	// For OR: if ANY child can't be parsed we must treat the whole OR as TRUE (no pruning)
	if (op == " AND ") {
		// If no children could be parsed, return TRUE
		if (child_conditions.empty()) {
			return "1=1"; // TRUE
		}
		// If only some children parsed, combine them with TRUE for missing parts
		if (child_conditions.size() < conj_expr.children.size()) {
			// Add TRUE for each missing child
			for (idx_t i = child_conditions.size(); i < conj_expr.children.size(); i++) {
				child_conditions.push_back("1=1");
			}
		}
	} else if (op == " OR ") {
		if (child_conditions.size() != conj_expr.children.size()) {
			// At least one child missing -> unknown OR part => cannot safely prune
			return "1=1"; // TRUE
		}
	}

	if (child_conditions.size() == 1) {
		return child_conditions[0];
	}

	return "(" + StringUtil::Join(child_conditions, op) + ")";
}

string GenerateComparisonCondition(const Expression &expr, const vector<column_t> &column_ids,
                                   const vector<idx_t> &field_ids) {
	// Check if this is a comparison expression
	if (expr.expression_class != ExpressionClass::BOUND_COMPARISON) {
		return string(); // Not a comparison we can handle
	}

	const auto &comp_expr = expr.Cast<BoundComparisonExpression>();

	// Check if we have a column reference and a constant
	BoundColumnRefExpression *column_ref = nullptr;
	BoundConstantExpression *constant_expr = nullptr;
	bool left_is_column = false;

	if (comp_expr.left->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
	    comp_expr.right->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
		column_ref = &comp_expr.left->Cast<BoundColumnRefExpression>();
		constant_expr = &comp_expr.right->Cast<BoundConstantExpression>();
		left_is_column = true;
	} else if (comp_expr.right->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
	           comp_expr.left->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
		column_ref = &comp_expr.right->Cast<BoundColumnRefExpression>();
		constant_expr = &comp_expr.left->Cast<BoundConstantExpression>();
		left_is_column = false;
	} else {
		// Unsupported comparison format
		return string();
	}

	// Get the column info
	auto column_idx = column_ref->binding.column_index;
	if (column_idx >= column_ids.size() || IsVirtualColumn(column_ids[column_idx])) {
		// Skip virtual columns for now
		return string();
	}

	// Get comparison type and adjust if column/constant order is reversed
	ExpressionType comp_type = comp_expr.type;
	if (!left_is_column) {
		// If constant is on left, flip the comparison
		switch (comp_type) {
		case ExpressionType::COMPARE_LESSTHAN:
			comp_type = ExpressionType::COMPARE_GREATERTHAN;
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			comp_type = ExpressionType::COMPARE_GREATERTHANOREQUALTO;
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			comp_type = ExpressionType::COMPARE_LESSTHAN;
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			comp_type = ExpressionType::COMPARE_LESSTHANOREQUALTO;
			break;
		default:
			// EQUAL and NOT_EQUAL don't need flipping
			break;
		}
	}

	// Map projected column position to actual table field id used in stats table
	auto column_id = field_ids[column_idx];

	// Create a ConstantFilter and use the existing logic to generate the min/max condition
	ConstantFilter constant_filter(comp_type, constant_expr->value);
	unordered_set<string> referenced_stats;
	string min_max_condition = GenerateConstantFilter(constant_filter, constant_expr->return_type, referenced_stats);

	if (min_max_condition.empty()) {
		return string();
	}

	// Generate the EXISTS subquery for this column
	string null_check;
	for (auto &stats_name : referenced_stats) {
		if (!null_check.empty()) {
			null_check += " OR ";
		}
		null_check += stats_name + " IS NULL";
	}

	string final_condition =
	    StringUtil::Format("column_id = %d AND (%s OR (%s))", column_id, null_check, min_max_condition);

	return StringUtil::Format("EXISTS (SELECT 1 FROM {METADATA_CATALOG}.ducklake_file_column_stats s WHERE "
	                          "s.data_file_id = base.data_file_id AND %s)",
	                          final_condition);
}

unique_ptr<MultiFileList>
DuckLakeMultiFileList::DynamicFilterPushdown(ClientContext &context, const MultiFileOptions &options,
                                             const vector<string> &names, const vector<LogicalType> &types,
                                             const vector<column_t> &column_ids, TableFilterSet &filters) const {
	if (read_info.scan_type != DuckLakeScanType::SCAN_TABLE) {
		// filter pushdown is only supported when scanning full tables
		return nullptr;
	}
	string filter = GenerateSimpleDynamicFilterPushDownQuery(column_ids, filters);
	if (!filter.empty()) {
		return make_uniq<DuckLakeMultiFileList>(read_info, transaction_local_files, transaction_local_data,
		                                        std::move(filter));
	}
	return nullptr;
}

vector<OpenFileInfo> DuckLakeMultiFileList::GetAllFiles() {
	vector<OpenFileInfo> file_list;
	for (idx_t i = 0; i < GetTotalFileCount(); i++) {
		file_list.push_back(GetFile(i));
	}
	return file_list;
}

FileExpandResult DuckLakeMultiFileList::GetExpandResult() {
	return FileExpandResult::MULTIPLE_FILES;
}

idx_t DuckLakeMultiFileList::GetTotalFileCount() {
	return GetFiles().size();
}

unique_ptr<NodeStatistics> DuckLakeMultiFileList::GetCardinality(ClientContext &context) {
	auto stats = read_info.table.GetTableStats(context);
	if (!stats) {
		return nullptr;
	}
	return make_uniq<NodeStatistics>(stats->record_count);
}

DuckLakeTableEntry &DuckLakeMultiFileList::GetTable() {
	return read_info.table;
}

OpenFileInfo DuckLakeMultiFileList::GetFile(idx_t i) {
	auto &files = GetFiles();
	if (i >= files.size()) {
		return OpenFileInfo();
	}
	auto &file_entry = files[i];
	auto &file = file_entry.file;
	OpenFileInfo result(file.path);
	auto extended_info = make_shared_ptr<ExtendedOpenFileInfo>();
	idx_t inlined_data_file_start = files.size() - inlined_data_tables.size();
	if (transaction_local_data) {
		inlined_data_file_start--;
	}
	if (transaction_local_data && i + 1 == files.size()) {
		// scanning transaction local data
		extended_info->options["transaction_local_data"] = Value::BOOLEAN(true);
		extended_info->options["inlined_data"] = Value::BOOLEAN(true);
		if (file_entry.row_id_start.IsValid()) {
			extended_info->options["row_id_start"] = Value::UBIGINT(file_entry.row_id_start.GetIndex());
		}
		extended_info->options["snapshot_id"] = Value(LogicalType::BIGINT);
		if (file_entry.mapping_id.IsValid()) {
			extended_info->options["mapping_id"] = Value::UBIGINT(file_entry.mapping_id.index);
		}
	} else if (i >= inlined_data_file_start) {
		// scanning inlined data
		auto inlined_data_index = i - inlined_data_file_start;
		auto &inlined_data_table = inlined_data_tables[inlined_data_index];
		extended_info->options["table_name"] = inlined_data_table.table_name;
		extended_info->options["inlined_data"] = Value::BOOLEAN(true);
		extended_info->options["schema_version"] =
		    Value::BIGINT(NumericCast<int64_t>(inlined_data_table.schema_version));
	} else {
		extended_info->options["file_size"] = Value::UBIGINT(file.file_size_bytes);
		if (file.footer_size.IsValid()) {
			extended_info->options["footer_size"] = Value::UBIGINT(file.footer_size.GetIndex());
		}
		if (files[i].row_id_start.IsValid()) {
			extended_info->options["row_id_start"] = Value::UBIGINT(files[i].row_id_start.GetIndex());
		}
		Value snapshot_id;
		if (files[i].snapshot_id.IsValid()) {
			snapshot_id = Value::BIGINT(NumericCast<int64_t>(files[i].snapshot_id.GetIndex()));
		} else {
			snapshot_id = Value(LogicalType::BIGINT);
		}
		extended_info->options["snapshot_id"] = std::move(snapshot_id);
		if (!file.encryption_key.empty()) {
			extended_info->options["encryption_key"] = Value::BLOB_RAW(file.encryption_key);
		}
		// files managed by DuckLake are never modified - we can keep them cached
		extended_info->options["validate_external_file_cache"] = Value::BOOLEAN(false);
		// etag / last modified time can be set to dummy values
		extended_info->options["etag"] = Value("");
		extended_info->options["last_modified"] = Value::TIMESTAMP(timestamp_t(0));
		if (!file_entry.delete_file.path.empty() || file_entry.max_row_count.IsValid()) {
			extended_info->options["has_deletes"] = Value::BOOLEAN(true);
		}
		if (file_entry.mapping_id.IsValid()) {
			extended_info->options["mapping_id"] = Value::UBIGINT(file_entry.mapping_id.index);
		}
	}
	result.extended_info = std::move(extended_info);
	return result;
}

unique_ptr<MultiFileList> DuckLakeMultiFileList::Copy() {
	auto result = make_uniq<DuckLakeMultiFileList>(read_info, transaction_local_files, transaction_local_data, filter);
	result->files = GetFiles();
	result->read_file_list = read_file_list;
	result->delete_scans = delete_scans;
	return std::move(result);
}

const DuckLakeFileListEntry &DuckLakeMultiFileList::GetFileEntry(idx_t file_idx) {
	auto &files = GetFiles();
	return files[file_idx];
}

DuckLakeFileData GetFileData(const DuckLakeDataFile &file) {
	DuckLakeFileData result;
	result.path = file.file_name;
	result.encryption_key = file.encryption_key;
	result.file_size_bytes = file.file_size_bytes;
	result.footer_size = file.footer_size;
	return result;
}

DuckLakeFileData GetDeleteData(const DuckLakeDataFile &file) {
	DuckLakeFileData result;
	if (!file.delete_file) {
		return result;
	}
	auto &delete_file = *file.delete_file;
	result.path = delete_file.file_name;
	result.encryption_key = delete_file.encryption_key;
	result.file_size_bytes = delete_file.file_size_bytes;
	result.footer_size = delete_file.footer_size;
	return result;
}

vector<DuckLakeFileListExtendedEntry> DuckLakeMultiFileList::GetFilesExtended() {
	lock_guard<mutex> l(file_lock);
	vector<DuckLakeFileListExtendedEntry> result;
	auto transaction_ref = read_info.GetTransaction();
	auto &transaction = *transaction_ref;
	if (!read_info.table_id.IsTransactionLocal()) {
		// not a transaction local table - read the file list from the metadata store
		auto &metadata_manager = transaction.GetMetadataManager();
		result = metadata_manager.GetExtendedFilesForTable(read_info.table, read_info.snapshot, filter);
	}
	if (transaction.HasDroppedFiles()) {
		for (idx_t file_idx = 0; file_idx < result.size(); file_idx++) {
			if (transaction.FileIsDropped(result[file_idx].file.path)) {
				result.erase_at(file_idx);
				file_idx--;
			}
		}
	}
	// if the transaction has any local deletes - apply them to the file list
	if (transaction.HasLocalDeletes(read_info.table_id)) {
		for (auto &file_entry : result) {
			transaction.GetLocalDeleteForFile(read_info.table_id, file_entry.file.path, file_entry.delete_file);
		}
	}
	idx_t transaction_row_start = TRANSACTION_LOCAL_ID_START;
	for (auto &file : transaction_local_files) {
		DuckLakeFileListExtendedEntry file_entry;
		file_entry.file_id = DataFileIndex();
		file_entry.delete_file_id = DataFileIndex();
		file_entry.row_count = file.row_count;
		file_entry.file = GetFileData(file);
		file_entry.delete_file = GetDeleteData(file);
		file_entry.row_id_start = transaction_row_start;
		transaction_row_start += file.row_count;
		result.push_back(std::move(file_entry));
	}
	inlined_data_tables = read_info.table.GetInlinedDataTables();
	for (auto &table : inlined_data_tables) {
		DuckLakeFileListExtendedEntry file_entry;
		file_entry.file.path = table.table_name;
		file_entry.file_id = DataFileIndex();
		file_entry.delete_file_id = DataFileIndex();
		file_entry.row_count = 0;
		file_entry.row_id_start = 0;
		file_entry.data_type = DuckLakeDataType::INLINED_DATA;
		result.push_back(std::move(file_entry));
	}
	if (transaction_local_data) {
		// we have transaction local inlined data - create the dummy file entry
		DuckLakeFileListExtendedEntry file_entry;
		file_entry.file.path = DUCKLAKE_TRANSACTION_LOCAL_INLINED_FILENAME;
		file_entry.file_id = DataFileIndex();
		file_entry.delete_file_id = DataFileIndex();
		file_entry.row_count = transaction_local_data->data->Count();
		file_entry.row_id_start = transaction_row_start;
		file_entry.data_type = DuckLakeDataType::TRANSACTION_LOCAL_INLINED_DATA;
		result.push_back(std::move(file_entry));
	}
	if (!read_file_list) {
		// we have not read the file list yet - construct it from the extended file list
		for (auto &file : result) {
			DuckLakeFileListEntry file_entry;
			file_entry.file = file.file;
			file_entry.row_id_start = file.row_id_start;
			file_entry.delete_file = file.delete_file;
			files.emplace_back(std::move(file_entry));
		}
		read_file_list = true;
	}
	return result;
}

void DuckLakeMultiFileList::GetFilesForTable() {
	auto transaction_ref = read_info.GetTransaction();
	auto &transaction = *transaction_ref;
	if (!read_info.table_id.IsTransactionLocal()) {
		// not a transaction local table - read the file list from the metadata store
		auto &metadata_manager = transaction.GetMetadataManager();
		files = metadata_manager.GetFilesForTable(read_info.table, read_info.snapshot, filter);
	}
	if (transaction.HasDroppedFiles()) {
		for (idx_t file_idx = 0; file_idx < files.size(); file_idx++) {
			if (transaction.FileIsDropped(files[file_idx].file.path)) {
				files.erase_at(file_idx);
				file_idx--;
			}
		}
	}
	// if the transaction has any local deletes - apply them to the file list
	if (transaction.HasLocalDeletes(read_info.table_id)) {
		for (auto &file_entry : files) {
			transaction.GetLocalDeleteForFile(read_info.table_id, file_entry.file.path, file_entry.delete_file);
		}
	}
	idx_t transaction_row_start = TRANSACTION_LOCAL_ID_START;
	for (auto &file : transaction_local_files) {
		DuckLakeFileListEntry file_entry;
		file_entry.file = GetFileData(file);
		file_entry.row_id_start = transaction_row_start;
		file_entry.delete_file = GetDeleteData(file);
		file_entry.mapping_id = file.mapping_id;
		transaction_row_start += file.row_count;
		files.emplace_back(std::move(file_entry));
	}
	inlined_data_tables = read_info.table.GetInlinedDataTables();
	for (auto &table : inlined_data_tables) {
		DuckLakeFileListEntry file_entry;
		file_entry.file.path = table.table_name;
		file_entry.row_id_start = 0;
		file_entry.data_type = DuckLakeDataType::INLINED_DATA;
		files.push_back(std::move(file_entry));
	}
	if (transaction_local_data) {
		// we have transaction local inlined data - create the dummy file entry
		DuckLakeFileListEntry file_entry;
		file_entry.file.path = DUCKLAKE_TRANSACTION_LOCAL_INLINED_FILENAME;
		file_entry.row_id_start = transaction_row_start;
		file_entry.data_type = DuckLakeDataType::TRANSACTION_LOCAL_INLINED_DATA;
		files.push_back(std::move(file_entry));
	}
}

void DuckLakeMultiFileList::GetTableInsertions() {
	if (read_info.table_id.IsTransactionLocal()) {
		throw InternalException("Cannot get changes between snapshots for transaction-local files");
	}
	auto transaction_ref = read_info.GetTransaction();
	auto &transaction = *transaction_ref;
	auto &metadata_manager = transaction.GetMetadataManager();
	files = metadata_manager.GetTableInsertions(read_info.table, *read_info.start_snapshot, read_info.snapshot);
	// add inlined data tables as sources (if any)
	inlined_data_tables = read_info.table.GetInlinedDataTables();
	for (auto &table : inlined_data_tables) {
		DuckLakeFileListEntry file_entry;
		file_entry.file.path = table.table_name;
		file_entry.row_id_start = 0;
		file_entry.data_type = DuckLakeDataType::INLINED_DATA;
		files.push_back(std::move(file_entry));
	}
}

void DuckLakeMultiFileList::GetTableDeletions() {
	if (read_info.table_id.IsTransactionLocal()) {
		throw InternalException("Cannot get changes between snapshots for transaction-local files");
	}
	auto transaction_ref = read_info.GetTransaction();
	auto &transaction = *transaction_ref;
	auto &metadata_manager = transaction.GetMetadataManager();
	delete_scans = metadata_manager.GetTableDeletions(read_info.table, *read_info.start_snapshot, read_info.snapshot);
	for (auto &file : delete_scans) {
		DuckLakeFileListEntry file_entry;
		file_entry.file = file.file;
		file_entry.row_id_start = file.row_id_start;
		file_entry.snapshot_id = file.snapshot_id;
		file_entry.mapping_id = file.mapping_id;
		files.emplace_back(std::move(file_entry));
	}
	// add inlined data tables as sources (if any)
	inlined_data_tables = read_info.table.GetInlinedDataTables();
	for (auto &table : inlined_data_tables) {
		DuckLakeFileListEntry file_entry;
		file_entry.file.path = table.table_name;
		file_entry.row_id_start = 0;
		file_entry.data_type = DuckLakeDataType::INLINED_DATA;
		files.push_back(std::move(file_entry));
	}
}

bool DuckLakeMultiFileList::IsDeleteScan() const {
	return read_info.scan_type == DuckLakeScanType::SCAN_DELETIONS;
}

const DuckLakeDeleteScanEntry &DuckLakeMultiFileList::GetDeleteScanEntry(idx_t file_idx) {
	return delete_scans[file_idx];
}

const vector<DuckLakeFileListEntry> &DuckLakeMultiFileList::GetFiles() {
	lock_guard<mutex> l(file_lock);
	if (!read_file_list) {
		// we have not read the file list yet - read it
		switch (read_info.scan_type) {
		case DuckLakeScanType::SCAN_TABLE:
			GetFilesForTable();
			break;
		case DuckLakeScanType::SCAN_INSERTIONS:
			GetTableInsertions();
			break;
		case DuckLakeScanType::SCAN_DELETIONS:
			GetTableDeletions();
			break;
		default:
			throw InternalException("Unknown DuckLake scan type");
		}
		read_file_list = true;
	}
	return files;
}

} // namespace duckdb
