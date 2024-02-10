// room_temp_graph.dart

import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:intl/intl.dart'; // Import DateFormat

class RoomTempGraph extends StatefulWidget {
  const RoomTempGraph({super.key});

  @override
  _RoomTempGraphState createState() => _RoomTempGraphState();
}

class _RoomTempGraphState extends State<RoomTempGraph> {
  Future<List<FlSpot>> fetchRoomTempData() async {
    final response = await http.get(Uri.parse('http://10.0.0.209:5001/api/graphs/room'));
    if (response.statusCode == 200) {
      final responseBody = jsonDecode(response.body);
      List<dynamic> roomTempData = responseBody['temperature'];
      List<FlSpot> spots = roomTempData.map((dataTuple) {
        double x = dataTuple[0]; // Use directly as it's already a floating-point number
        double y = dataTuple[1];
        return FlSpot(x, y);
      }).toList();
      return spots;
    } else {
      throw Exception('Failed to load temperature data');
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<FlSpot>>(
      future: fetchRoomTempData(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          if (snapshot.hasData) {
            return LineChart(
              LineChartData(
                gridData: FlGridData(show: true),
                titlesData: FlTitlesData(
                  bottomTitles: SideTitles(
                    showTitles: true,
                    getTitles: (value) {
                      // Convert Unix timestamp to a Date object
                      var dateTime = DateTime.fromMillisecondsSinceEpoch(value.toInt() * 1000);
                      // Format the dateTime to include hours, minutes, and seconds
                      return DateFormat('HH:mm:ss').format(dateTime); // "HH" for 24-hour format
                    },
                    reservedSize: 36, // Adjust based on your layout needs
                  ),
                  // Configure other title properties as needed
                ),
                borderData: FlBorderData(show: true),
                lineBarsData: [
                  LineChartBarData(spots: snapshot.data!),
                ],
              ),
            );
          } else if (snapshot.hasError) {
            return Center(child: Text("Error loading graph data: ${snapshot.error}"));
          }
        }
        // By default, show a loading spinner.
        return const Center(child: CircularProgressIndicator());
      },
    );
  }
}
