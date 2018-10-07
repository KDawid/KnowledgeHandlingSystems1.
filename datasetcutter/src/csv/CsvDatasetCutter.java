package csv;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class CsvDatasetCutter {
    public static void main( String[] args ) {
    	if(args.length<3) {
    		System.out.println("Mandatory parameters: \n"
    				+ "\t1. Location of the file\n"
    				+ "\t2. Output file location\n"
    				+ "\t3. Megabytes");
    		return;
    	}
    	CsvDatasetCutter cutter = new CsvDatasetCutter();
    	cutter.copy(args[0], args[1], Integer.parseInt(args[2]));
        System.out.println( "Done. Don't forget to check the end of the file - it is probably NOT a valid csv." );
    }

	private void copy(String sourceFile, String destFile, int kiloBytes) {
	    FileInputStream fis = null;
	    FileOutputStream fos = null;
	    
	    try {
	        fis = new FileInputStream(sourceFile);
	        fos = new FileOutputStream(destFile);
	         
	        byte[] buffer = new byte[1024];
	        int noOfBytes = 0;
	 
	        System.out.println("Copying file using streams. Size: " + kiloBytes + " kilobytes.");
	 
	        // read bytes from source file and write to destination file
	        int i=0;
	        while ((noOfBytes = fis.read(buffer)) != -1 && i<kiloBytes) {
	         	i++;
	            fos.write(buffer, 0, noOfBytes);
	        }
	    }
	    catch (FileNotFoundException e) {
	        System.out.println("File not found" + e);
	    }
	    catch (IOException ioe) {
	        System.out.println("Exception while copying file " + ioe);
	    }
	    finally {
	        // close the streams using close method
	        try {
	            if (fis != null) {
	                fis.close();
	            }
	            if (fos != null) {
	                fos.close();
	            }
	        }
	        catch (IOException ioe) {
	            System.out.println("Error while closing stream: " + ioe);
	        }
	    }
	}
}
