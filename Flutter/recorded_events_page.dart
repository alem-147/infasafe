import 'package:flutter/material.dart';

class RecordedEventsPage extends StatelessWidget {
  const RecordedEventsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Events'),
      ),
      body: const Center(
        child: Text('This is the recorded events page'),
      ),
    );
  }
}