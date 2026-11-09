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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val DashboardBackground = Color(0xFF08111F)
private val DashboardCard = Color(0xFF111F32)
private val DashboardAccent = Color(0xFF00E5A8)

private fun isSafeLabel(label: String): Boolean {
    val value = label.trim().lowercase(Locale.ROOT)
    return value in setOf(
        "likely safe",
        "safe",
        "benign",
        "likely normal traffic",
        "normal traffic",
        "normal"
    )
}

private fun isThreatLabel(label: String): Boolean {
    val value = label.trim().lowercase(Locale.ROOT)

    return value.contains("scam") ||
        value.contains("phishing") ||
        value.contains("suspicious") ||
        value.contains("potential threat") ||
        value.contains("dangerous") ||
        value.contains("fraud") ||
        value.contains("malicious") ||
        value.contains("unsafe")
}

@Composable
fun DashboardScreen(
    onHistory: () -> Unit,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val repository = remember { LearningRepository(context) }
    val scope = rememberCoroutineScope()

    var records by remember { mutableStateOf<List<ScanRecord>>(emptyList()) }
    var loading by remember { mutableStateOf(true) }
    var confirmClear by remember { mutableStateOf(false) }

    suspend fun refreshDashboard() {
        loading = true
        records = repository.getAllScans()
        loading = false
    }

    LaunchedEffect(Unit) {
        refreshDashboard()
    }

    val total = records.size
    val safeCount = records.count { isSafeLabel(it.resultLabel) }
    val threatCount = records.count { isThreatLabel(it.resultLabel) }
    val unknownCount = total - safeCount - threatCount

    val detectionRate = if (total > 0) {
        threatCount * 100.0 / total
    } else {
        0.0
    }

    val smsCount = records.count { it.analyzer.equals("SMS", true) }
    val urlCount = records.count { it.analyzer.equals("URL", true) }
    val networkCount = records.count { it.analyzer.equals("NETWORK", true) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(DashboardBackground)
            .padding(18.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            TextButton(onClick = onBack) {
                Text("← Back", color = DashboardAccent)
            }
            Spacer(Modifier.weight(1f))
            Text("DASHBOARD 2.0", color = DashboardAccent, fontSize = 19.sp)
        }

        Spacer(Modifier.height(12.dp))

        if (loading) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(color = DashboardAccent)
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    Text(
                        "Your Security Overview",
                        color = Color.White,
                        fontSize = 24.sp
                    )
                }

                item {
                    DashboardStatCard(
                        "Total Scans",
                        total.toString(),
                        DashboardAccent
                    )
                }

                item {
                    DashboardStatCard(
                        "Safe Results",
                        safeCount.toString(),
                        Color(0xFF4CAF50)
                    )
                }

                item {
                    DashboardStatCard(
                        "Threats Detected",
                        threatCount.toString(),
                        Color(0xFFFF5252)
                    )
                }

                item {
                    DashboardStatCard(
                        "Threat Detection Rate",
                        String.format(Locale.US, "%.1f%%", detectionRate),
                        Color(0xFFFFC107)
                    )
                }

                item {
                    DashboardStatCard(
                        "Unclassified Results",
                        unknownCount.toString(),
                        Color.LightGray
                    )
                }

                item {
                    Text(
                        "Analyzer Breakdown",
                        color = Color.White,
                        fontSize = 20.sp
                    )
                }

                item {
                    DashboardStatCard("SMS Scans", smsCount.toString(), DashboardAccent)
                }

                item {
                    DashboardStatCard("URL Scans", urlCount.toString(), DashboardAccent)
                }

                item {
                    DashboardStatCard(
                        "Network Scans",
                        networkCount.toString(),
                        DashboardAccent
                    )
                }

                item {
                    Button(
                        onClick = onHistory,
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = DashboardAccent,
                            contentColor = DashboardBackground
                        )
                    ) {
                        Text("VIEW SCAN HISTORY")
                    }
                }

                item {
                    OutlinedButton(
                        onClick = { confirmClear = true },
                        enabled = records.isNotEmpty(),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("CLEAR ALL SCAN HISTORY", color = Color(0xFFFF5252))
                    }
                }

                item {
                    Text("Recent Activity", color = Color.White, fontSize = 20.sp)
                }

                if (records.isEmpty()) {
                    item {
                        Text(
                            "No scans saved yet. Analyze an SMS, URL, or network record.",
                            color = Color.LightGray
                        )
                    }
                } else {
                    items(
                        records.sortedByDescending { it.timestamp }.take(5),
                        key = { it.id }
                    ) { record ->
                        ScanHistoryItem(record)
                    }
                }
            }
        }
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Clear scan history?") },
            text = {
                Text(
                    "This deletes all $total scan records. " +
                        "Your saved feedback and detector models will remain unchanged."
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmClear = false
                        scope.launch {
                            repository.deleteAllScans()
                            refreshDashboard()
                        }
                    }
                ) {
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

@Composable
private fun DashboardStatCard(
    title: String,
    value: String,
    color: Color
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DashboardCard)
    ) {
        Row(
            modifier = Modifier
                .padding(18.dp)
                .fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(title, color = Color.White, fontSize = 16.sp)
            Spacer(Modifier.weight(1f))
            Text(value, color = color, fontSize = 24.sp)
        }
    }
}

@Composable
fun ScanHistoryItem(record: ScanRecord) {
    val date = remember(record.timestamp) {
        SimpleDateFormat("dd MMM yyyy, hh:mm a", Locale.getDefault())
            .format(Date(record.timestamp))
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = DashboardCard)
    ) {
        Column(Modifier.padding(15.dp)) {
            Text(record.analyzer, color = DashboardAccent, fontSize = 17.sp)
            Spacer(Modifier.height(5.dp))
            Text(record.resultLabel, color = Color.White, fontSize = 16.sp)
            Text(
                "Risk score: ${(record.riskScore * 100).toInt()}%",
                color = Color.LightGray
            )
            Text(date, color = Color.Gray, fontSize = 12.sp)
        }
    }
}
