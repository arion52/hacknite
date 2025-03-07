import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class GraphCard extends StatelessWidget {
  final String title;
  final List<Map<String, dynamic>> data;

  const GraphCard({super.key, required this.title, required this.data});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: const Color(0xFF212121),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.purple.withOpacity(0.5),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Container(
            height: MediaQuery.of(context).size.width / 2.5,
            padding: const EdgeInsets.all(8.0),
            decoration: BoxDecoration(
              color: Colors.grey[800],
              borderRadius: BorderRadius.circular(15),
            ),
            child: data.isEmpty
                ? const Center(
                    child: Text('No Data', style: TextStyle(color: Colors.white70)),
                  )
                : LineChart(
                    LineChartData(
                      backgroundColor: Colors.grey[800],
                      gridData: FlGridData(show: false),
                      borderData: FlBorderData(
                        show: true,
                        border: Border.all(color: Colors.white70),
                      ),
                      titlesData: FlTitlesData(
                        leftTitles: AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            reservedSize: 30,
                            getTitlesWidget: (value, meta) => Text(
                              '${value.toInt()} kW',
                              style: const TextStyle(color: Colors.white, fontSize: 10),
                            ),
                          ),
                        ),
                        bottomTitles: AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            getTitlesWidget: (value, meta) {
                              int index = value.toInt();
                              if (index < 0 || index >= data.length) return const SizedBox();
                              return Text(
                                data[index]['label'],
                                style: const TextStyle(color: Colors.white, fontSize: 10),
                              );
                            },
                          ),
                        ),
                      ),
                      lineBarsData: [
                        LineChartBarData(
                          spots: data.asMap().entries.map((entry) {
                            int index = entry.key;
                            double value = double.tryParse(entry.value['value'].split(' ')[0]) ?? 0;
                            return FlSpot(index.toDouble(), value);
                          }).toList(),
                          isCurved: true,
                          color: Colors.purple,
                          barWidth: 3,
                          isStrokeCapRound: true,
                          belowBarData: BarAreaData(show: true, color: Colors.purple.withOpacity(0.2)),
                        ),
                      ],
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}
