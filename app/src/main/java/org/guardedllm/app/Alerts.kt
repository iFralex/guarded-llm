package org.guardedllm.app

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat

object Alerts {
    fun show(context: Context) {
        if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(context, android.Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) return
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel("research_alerts", "Research alerts", NotificationManager.IMPORTANCE_DEFAULT))
        val pending = PendingIntent.getActivity(context, 0, Intent(context, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        manager.notify(1, NotificationCompat.Builder(context, "research_alerts")
            .setSmallIcon(android.R.drawable.ic_dialog_alert).setContentTitle("Review a conversation")
            .setContentText("Potentially harmful language was detected. Consider the context.")
            .setContentIntent(pending).setAutoCancel(true).build())
    }
}
