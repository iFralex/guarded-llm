package org.guardedllm.app

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { AppScreen() } }
    }
}

@Composable fun AppScreen() {
    val context = LocalContext.current
    val mode by Monitoring.mode.collectAsStateWithLifecycle()
    val status by Monitoring.status.collectAsStateWithLifecycle()
    val result by Monitoring.result.collectAsStateWithLifecycle()
    var text by remember { mutableStateOf("") }
    var disclosure by remember { mutableStateOf(false) }
    val notificationPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
    Scaffold { padding -> Column(Modifier.fillMaxSize().padding(padding).verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Guarded LLM", style = MaterialTheme.typography.headlineMedium)
        Text("Research prototype · English only", style = MaterialTheme.typography.titleMedium)
        Text("A toxicity model can make mistakes. It does not detect or rule out grooming, bullying, or danger.")
        HorizontalDivider()
        Text("Try text locally", style = MaterialTheme.typography.titleLarge)
        OutlinedTextField(value = text, onValueChange = { text = it.take(4096) }, label = { Text("Text to analyze") }, modifier = Modifier.fillMaxWidth(), minLines = 3)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { Monitoring.analyzeManual(context, text) }) { Text("Analyze") }
            OutlinedButton(onClick = { text = "" }) { Text("Clear text") }
        }
        Text(status)
        result?.let { output ->
            Text("Toxicity score: ${"%.2f".format(java.util.Locale.US, output.scores["toxic"] ?: 0f)} · ${output.elapsedMs} ms")
            Text(if (output.flagged) "Consider reviewing the conversation in context." else "A low score does not establish safety.")
        }
        HorizontalDivider()
        Text("Instagram monitoring", style = MaterialTheme.typography.titleLarge)
        Text("Status: ${mode.name.lowercase().replaceFirstChar { it.uppercase() }}")
        if (mode == Monitoring.Mode.STOPPED) Button(onClick = { disclosure = true }) { Text("Enable monitoring") }
        if (mode == Monitoring.Mode.ACTIVE) OutlinedButton(onClick = { Monitoring.setMode(Monitoring.Mode.PAUSED) }) { Text("Pause") }
        if (mode == Monitoring.Mode.PAUSED) Button(onClick = { Monitoring.setMode(Monitoring.Mode.ACTIVE) }) { Text("Resume") }
        if (mode != Monitoring.Mode.STOPPED) OutlinedButton(onClick = { Monitoring.setMode(Monitoring.Mode.STOPPED) }) { Text("Stop monitoring") }
        Text("Instagram may not expose chat messages to accessibility services. Detection is experimental and incomplete.")
    } }
    if (disclosure) AlertDialog(onDismissRequest = { disclosure = false }, title = { Text("Access to Instagram content") }, text = {
        Text("When enabled, this service reads visible text identified as chat messages in Instagram and analyzes it on this device. It may see sensitive content. Conversation text is not saved or sent to a server. Only enable this for your own device and test conversations. Monitoring can be paused or stopped here. Android will ask you to enable the accessibility service in Settings.")
    }, confirmButton = { TextButton(onClick = {
        disclosure = false
        Monitoring.setMode(Monitoring.Mode.ACTIVE)
        if (Build.VERSION.SDK_INT >= 33) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
    }) { Text("I agree and continue") } }, dismissButton = { TextButton(onClick = { disclosure = false }) { Text("Cancel") } })
}
