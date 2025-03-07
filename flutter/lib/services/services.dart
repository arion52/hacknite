import 'dart:convert';
import 'package:http/http.dart' as http;

// Base URL (adjust if your IP changes)
const String baseUrl = 'http://172.16.45.187:8000';

Future<List<Map<String, dynamic>>> fetchGenerationBatch(String period) async {
  final response = await http.get(Uri.parse('$baseUrl/predict_generation_batch/?period=$period'));

  if (response.statusCode == 200) {
    final responseData = json.decode(response.body);
    final predictions = responseData['predictions'] as List<dynamic>;

    return predictions.map<Map<String, dynamic>>((item) {
      return {
        'label': _formatTimestamp(item['timestamp']),
        'value': '${item['generation']} kW',
      };
    }).toList();
  } else {
    throw Exception('Failed to fetch generation batch data');
  }
}

String _formatTimestamp(String timestamp) {
  final dateTime = DateTime.parse(timestamp);
  return 'Hour ${dateTime.hour}';
}


