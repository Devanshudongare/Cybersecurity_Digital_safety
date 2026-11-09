package com.devanshu.cyberguardai

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Delete

@Dao
interface LearningDao {

    // Existing feedback operations
    @Insert
    suspend fun insert(record: LearningRecord)

    @Query("SELECT * FROM learning_records ORDER BY timestamp DESC")
    suspend fun getAll(): List<LearningRecord>

    @Query("SELECT * FROM learning_records WHERE analyzer = :type ORDER BY timestamp DESC")
    suspend fun getByAnalyzer(type: String): List<LearningRecord>

    @Query("SELECT COUNT(*) FROM learning_records")
    suspend fun count(): Int

    @Delete
    suspend fun delete(record: LearningRecord)

    @Query("DELETE FROM learning_records")
    suspend fun deleteAll()

    // New scan-history operations
    @Insert
    suspend fun insertScan(record: ScanRecord)

    @Query("SELECT * FROM scan_history ORDER BY timestamp DESC")
    suspend fun getAllScans(): List<ScanRecord>

    @Query("SELECT COUNT(*) FROM scan_history")
    suspend fun scanCount(): Int

    @Query("SELECT COUNT(*) FROM scan_history WHERE resultLabel = :label")
    suspend fun countScansByLabel(label: String): Int

    @Delete
    suspend fun deleteScan(record: ScanRecord)

    @Query("DELETE FROM scan_history")
    suspend fun deleteAllScans()
}
