// graphs_page.dart

import 'package:flutter/material.dart';
import 'room_temp_graph.dart'; // Import the temperature graph widget
import 'room_rh_graph.dart'; // Import the temperature graph widget

class GraphsPage extends StatelessWidget {
  const GraphsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Room Data Graphs'),
      ),
      body: const SingleChildScrollView( // Allows the content to be scrollable
        child: Padding(
          padding: EdgeInsets.all(8.0),
          child: Column(
            children: [
              SizedBox(
                height: 300, // Specify the height for the chart
                child: RoomTempGraph(), // Temperature graph widget
              ),
              SizedBox(height: 20), // Spacing between the graphs
              SizedBox(
                height: 300, // Specify the height for the chart
                child: RoomRHGraph(), // Humidity graph widget
              ),
            ],
          ),
        ),
      ),
    );
  }
}
