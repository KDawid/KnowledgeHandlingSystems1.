package csv;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;

import au.com.bytecode.opencsv.CSVReader;

public class CsvToJsonConverter {

	public static void main(String[] args) {
		if (args.length < 1) {
			System.out.println("Please add the input file path");
			return;
		}
		CsvToJsonConverter converter = new CsvToJsonConverter();
		converter.run(args[0]);
		System.out.println("end.");
	}

	public void run(String file) {
		try (CSVReader reader = new CSVReader(new FileReader(file))) {
			writeJson(reader);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void writeJson(CSVReader reader) throws IOException {
		JsonFactory fac = new JsonFactory();
		JsonGenerator gen = fac.createGenerator(new File("result_java.json"), JsonEncoding.UTF8)
				.useDefaultPrettyPrinter();
		String[] tokens;
		String[] headers = reader.readNext();
		gen.writeStartArray();
		while ((tokens = reader.readNext()) != null) {
			gen.writeStartObject();
			for (int i = 0; i < headers.length; i++) {
				String value = i < tokens.length ? tokens[i] : null;
				gen.writeStringField(headers[i], value);
			}
			gen.writeEndObject();
		}
		gen.writeEndArray();
	}
}
