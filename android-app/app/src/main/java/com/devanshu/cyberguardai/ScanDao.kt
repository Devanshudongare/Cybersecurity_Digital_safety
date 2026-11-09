package com.devanshu.cyberguardai

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Delete

@Dao
interface ScanDao {

    @Insert
    suspend fun insert(record: ScanRecord)

    @Query("SELECT * FROM scan_history ORDER BY timestamp DESC")
    suspend fun getAll(): List<ScanRecord>

    @Query("SELECT COUNT(*) FROM scan_history")
    suspend fun count(): Int

    @Query("SELECT COUNT(*) FROM scan_history WHERE result = :label")
    suspend fun countByResult(label: String): Int

    @Query("SELECT * FROM scan_history WHERE id = :recordId LIMIT 1")
    suspend fun getById(recordId: Int): ScanRecord?

    @Delete
    suspend fun delete(record: ScanRecord)

    @Query("DELETE FROM scan_history")
    suspend fun deleteAll()
}
