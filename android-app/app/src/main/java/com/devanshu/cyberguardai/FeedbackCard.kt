package com.devanshu.cyberguardai

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.launch

@Composable
fun FeedbackCard(
    analyzer: String,
    inputData: String,
    predictedLabel: String,
    confidence: Double,
    possibleLabels: List<String>,
    language: AppLanguage = AppLanguage.ENGLISH
) {
    val context = LocalContext.current
    val repository = remember { LearningRepository(context) }
    val scope = rememberCoroutineScope()

    var showCorrection by remember { mutableStateOf(false) }
    var saving by remember { mutableStateOf(false) }
    var feedbackMessage by remember { mutableStateOf("") }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF111F32))
    ) {
        Column(Modifier.padding(18.dp)) {
            Text(
                language.translate("verify"),
                color = Color(0xFF00E5A8),
                fontSize = 20.sp
            )

            Spacer(Modifier.height(10.dp))

            Text(
                language.translate("is_prediction_correct"),
                color = Color.White,
                fontSize = 17.sp
            )

            Spacer(Modifier.height(12.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    modifier = Modifier.weight(1f),
                    enabled = !saving && feedbackMessage.isEmpty(),
                    onClick = {
                        saving = true
                        scope.launch {
                            try {
                                repository.saveVerifiedExample(
                                    analyzer,
                                    inputData,
                                    predictedLabel,
                                    predictedLabel,
                                    confidence
                                )
                                feedbackMessage = language.translate("saved")
                            } catch (e: Exception) {
                                feedbackMessage = language.translate("save_error")
                            }
                            saving = false
                        }
                    }
                ) {
                    Text("✓ ${language.translate("correct")}")
                }

                Button(
                    modifier = Modifier.weight(1f),
                    enabled = !saving && feedbackMessage.isEmpty(),
                    onClick = { showCorrection = true },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFB83232)
                    )
                ) {
                    Text("✕ ${language.translate("incorrect")}")
                }
            }

            if (showCorrection) {
                Spacer(Modifier.height(12.dp))

                Text(
                    language.translate("select_correct"),
                    color = Color.White
                )

                possibleLabels.filter { it != predictedLabel }.forEach { label ->
                    TextButton(
                        enabled = !saving && feedbackMessage.isEmpty(),
                        onClick = {
                            saving = true
                            scope.launch {
                                try {
                                    repository.saveVerifiedExample(
                                        analyzer,
                                        inputData,
                                        predictedLabel,
                                        label,
                                        confidence
                                    )
                                    feedbackMessage =
                                        language.translate("correction_saved")
                                    showCorrection = false
                                } catch (e: Exception) {
                                    feedbackMessage =
                                        language.translate("save_error")
                                }
                                saving = false
                            }
                        }
                    ) {
                        Text(
                            "${language.translate("actually")}: $label",
                            color = Color(0xFF00E5A8)
                        )
                    }
                }
            }

            if (saving) {
                Spacer(Modifier.height(8.dp))
                CircularProgressIndicator()
            }

            if (feedbackMessage.isNotEmpty()) {
                Spacer(Modifier.height(10.dp))
                Text(
                    feedbackMessage,
                    color = Color(0xFF00E5A8),
                    fontSize = 15.sp
                )
            }
        }
    }
}
