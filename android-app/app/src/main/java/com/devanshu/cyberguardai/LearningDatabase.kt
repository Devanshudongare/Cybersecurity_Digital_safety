package com.devanshu.cyberguardai

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [LearningRecord::class, ScanRecord::class],
    version = 2,
    exportSchema = false
)
abstract class LearningDatabase : RoomDatabase() {

    abstract fun learningDao(): LearningDao

    companion object {
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS scan_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        analyzer TEXT NOT NULL,
                        inputData TEXT NOT NULL,
                        resultLabel TEXT NOT NULL,
                        riskScore REAL NOT NULL,
                        timestamp INTEGER NOT NULL
                    )
                """.trimIndent())
            }
        }

        @Volatile
        private var INSTANCE: LearningDatabase? = null

        fun getDatabase(context: Context): LearningDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    LearningDatabase::class.java,
                    "cyberguard_learning.db"
                )
                    .addMigrations(MIGRATION_1_2)
                    .build()
                    .also { INSTANCE = it }
            }
        }
    }
}
