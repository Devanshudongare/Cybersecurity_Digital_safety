package com.devanshu.cyberguardai

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.launch

@Composable
fun HistoryScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val repository = remember { LearningRepository(context) }
    val scope = rememberCoroutineScope()

    var records by remember { mutableStateOf<List<ScanRecord>>(emptyList()) }
    var loading by remember { mutableStateOf(true) }
    var confirmClear by remember { mutableStateOf(false) }

    suspend fun refreshHistory() {
        records = repository.getAllScans()
        loading = false
    }

    LaunchedEffect(Unit) {
        refreshHistory()
    }

    Column(
        modifier = Modifier.fillMaxSize()
            .background(Color(0xFF08111F))
            .padding(18.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            TextButton(onClick = onBack) {
                Text("← Back", color = Color(0xFF00E5A8))
            }
            Spacer(Modifier.weight(1f))
            Text("SCAN HISTORY", color = Color(0xFF00E5A8), fontSize = 20.sp)
        }

        Spacer(Modifier.height(15.dp))

        Text(
            "${records.size} saved scans",
            color = Color.White,
            fontSize = 18.sp
        )

        Spacer(Modifier.height(10.dp))

        OutlinedButton(
            onClick = { confirmClear = true },
            enabled = records.isNotEmpty(),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("CLEAR ALL HISTORY", color = Color(0xFFFF5252))
        }

        Spacer(Modifier.height(15.dp))

        when {
            loading -> CircularProgressIndicator(color = Color(0xFF00E5A8))

            records.isEmpty() -> Text(
                "No scan history available.",
                color = Color.LightGray
            )

            else -> LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                items(records, key = { it.id }) { record ->
                    Column {
                        ScanHistoryItem(record)

                        TextButton(
                            onClick = {
                                scope.launch {
                                    repository.deleteScan(record)
                                    refreshHistory()
                                }
                            }
                        ) {
                            Text("Delete this scan", color = Color(0xFFFF5252))
                        }
                    }
                }
            }
        }
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear scan history?") },
            text = { Text("All saved scan records will be permanently deleted.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmClear = false
                    scope.launch {
                        repository.deleteAllScans()
                        refreshHistory()
                    }
                }) {
                    Text("Clear", color = Color.Red)
                }
            },
            dismissButton = {
                TextButton(onClick = { confirmClear = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}
