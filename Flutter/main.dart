import 'package:flutter/material.dart';
import 'live_stream_page.dart'; // Create this file for the live stream page
import 'recorded_events_page.dart'; // Create this file for the recorded events page
import 'graphs_page.dart'; // Create this file for the graphs page

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sensor Dashboard',
      initialRoute: '/',
      routes: {
        '/': (context) => const HomePage(),
        '/live_stream': (context) => const LiveStreamPage(), // Define this Widget in its file
        '/recorded_events': (context) => const RecordedEventsPage(), // Define this Widget in its file
        '/graphs': (context) => const GraphsPage(), // Define this Widget in its file
      },
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Home Page'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/live_stream');
              },
              child: const Text('Live Stream'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/recorded_events');
              },
              child: const Text('Recorded Events'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/graphs');
              },
              child: const Text('Graphs'),
            ),
          ],
        ),
      ),
    );
  }
}
