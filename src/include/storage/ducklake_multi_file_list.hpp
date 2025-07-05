//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_multi_file_list.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "storage/ducklake_scan.hpp"
#include "storage/ducklake_transaction.hpp"
#include "storage/ducklake_metadata_info.hpp"
#include "storage/ducklake_inlined_data.hpp"

namespace duckdb {

//! The DuckLakeMultiFileList implements the MultiFileList API to allow injecting it into the regular DuckDB parquet
//! scan
class DuckLakeMultiFileList : public MultiFileList {
	static constexpr const idx_t TRANSACTION_LOCAL_ID_START = 1000000000000000000ULL;
	static constexpr const char *DUCKLAKE_TRANSACTION_LOCAL_INLINED_FILENAME =
	    "__ducklake_inlined_transaction_local_data";

public:
	DuckLakeMultiFileList(DuckLakeFunctionInfo &read_info, vector<DuckLakeDataFile> transaction_local_files,
	                      shared_ptr<DuckLakeInlinedData> transaction_local_data, string filter = string());
	DuckLakeMultiFileList(DuckLakeFunctionInfo &read_info, vector<DuckLakeFileListEntry> files_to_scan);
	DuckLakeMultiFileList(DuckLakeFunctionInfo &read_info, const DuckLakeInlinedTableInfo &inlined_table);

	unique_ptr<MultiFileList> ComplexFilterPushdown(ClientContext &context, const MultiFileOptions &options,
	                                                MultiFilePushdownInfo &info,
	                                                vector<unique_ptr<Expression>> &filters) override;

	unique_ptr<MultiFileList> DynamicFilterPushdown(ClientContext &context, const MultiFileOptions &options,
	                                                const vector<string> &names, const vector<LogicalType> &types,
	                                                const vector<column_t> &column_ids,
	                                                TableFilterSet &filters) const override;

	vector<OpenFileInfo> GetAllFiles() override;
	FileExpandResult GetExpandResult() override;
	idx_t GetTotalFileCount() override;
	unique_ptr<NodeStatistics> GetCardinality(ClientContext &context) override;
	DuckLakeTableEntry &GetTable();
	unique_ptr<MultiFileList> Copy() override;
	bool HasTransactionLocalData() const {
		return !transaction_local_files.empty() || transaction_local_data;
	}
	vector<DuckLakeFileListExtendedEntry> GetFilesExtended();
	const vector<DuckLakeFileListEntry> &GetFiles();
	const DuckLakeFileListEntry &GetFileEntry(idx_t file_idx);

	bool IsDeleteScan() const;
	const DuckLakeDeleteScanEntry &GetDeleteScanEntry(idx_t file_idx);

protected:
	//! Get the i-th expanded file
	OpenFileInfo GetFile(idx_t i) override;

private:
	void GetFilesForTable();
	void GetTableInsertions();
	void GetTableDeletions();

private:
	mutex file_lock;
	DuckLakeFunctionInfo &read_info;
	//! The set of files to read
	vector<DuckLakeFileListEntry> files;
	bool read_file_list;
	//! The set of transaction-local files
	vector<DuckLakeDataFile> transaction_local_files;
	//! Inlined transaction-local data
	shared_ptr<DuckLakeInlinedData> transaction_local_data;
	//! Inlined data tables
	vector<DuckLakeInlinedTableInfo> inlined_data_tables;
	//! The set of delete scans, only used when scanning deleted tuples using ducklake_table_deletions
	vector<DuckLakeDeleteScanEntry> delete_scans;
	//! The filter to apply
	string filter;
};

} // namespace duckdb
