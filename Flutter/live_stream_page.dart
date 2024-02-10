import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

class LiveStreamPage extends StatefulWidget {
  const LiveStreamPage({Key? key}) : super(key: key);

  @override
  _LiveStreamPageState createState() => _LiveStreamPageState();
}

class _LiveStreamPageState extends State<LiveStreamPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Stream'),
      ),
      body: const WebView(
        initialUrl: 'http://10.0.0.209:8554/', // Replace <JETSON-IP> with your Jetson device's IP address
        javascriptMode: JavascriptMode.unrestricted, // Enable JavaScript for WebRTC
      ),
    );
  }
}