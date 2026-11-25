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

## Paired acquisition comparison

Use two test accounts on one physical Android phone, with the same Instagram version and settings for both passes. Run the prototype and the AWARE-Light screen-text sensor in separate passes. Record the exact framework version, configuration and whether the sensor is available and exposes a usable capture record. Keep capture files local and use invented text for this exercise.

Send the following messages in order. Preserve capitalization, punctuation and the identical repeated text.

| Turn | Sender | Text |
| --- | --- | --- |
| 1 | A | Are we still meeting after class? |
| 2 | B | Yes, meet me by the library. |
| 3 | A | I'll be there in five. |
| 4 | B | OK, I’ll wait. |
| 5 | B | OK, I’ll wait. |
| 6 | A | Nobody wants you in our group. |
| 7 | B | Please stop sending me that. |
| 8 | A | That's enough for today's test. |

For each pass, open the chat, receive the first four turns, briefly leave the chat, return for the remaining turns, scroll up and down, then switch to a non-chat screen and stop capture. Record which turns were visible and when. Keep an ordered record of what each pipeline actually receives. If the prototype only exposes accepted events, use a local debug inspection of those events and record that measurement point. Compare sensor output and classifier input separately when possible.

Match captures to reference turns in order, preserving the two intentional repetitions. Count missed visible turns, unintended duplicates, out-of-order turns, extra interface text and failures to distinguish sender direction. Record one comparison per run and include Android and Instagram versions. Feed the clean reference and each captured transcript to the same classifier with the same context and threshold, then record changes in score or decision. Treat this as a functional check on invented chat text.
