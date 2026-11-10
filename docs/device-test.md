# Physical-device verification record

Status: **not run**. Record Android version, phone model, Instagram version, APK version and model revision before testing. Use test accounts and synthetic conversations, and do not record real message text in logs or reports.

- Install APK and open offline. Confirm manual inference succeeds for harmless and obviously toxic test text.
- Decline disclosure: service must not process content. Accept disclosure and enable the system accessibility service. Grant notification permission when prompted.
- Open a test Instagram chat. Check whether the chat composer and message nodes are exposed with Android accessibility inspection. Record coverage, missing messages, duplication, latency, and whether sent/received direction can be identified. The implementation supports only explicitly named message nodes; unknown trees intentionally produce no analysis.
- Pause, navigate through chats, then resume; stop and verify no more analyses or alerts. Disable the system service and reopen the app. Restart the app process and verify monitoring is off.
- Turn on airplane mode and repeat manual and supported chat tests. Check the application has no INTERNET permission and notifications contain no message content.
- Test an unavailable/corrupt model, denied notification permission and rapid navigation. On Android 13+ alerts require notification permission.
- Record warm and cold inference latency, peak memory and APK/model size; state device and methodology. Do not infer battery usage or real-world accuracy from a short run.

None of these observations, accuracy metrics or latency results have been measured in this workspace.
