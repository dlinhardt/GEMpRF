import json
import os
import datetime

class JsonMgr:
    @classmethod
    def write_to_file(cls, filepath, data):

        # create folders     
        # Extract the directory path from the file path
        directory_path = os.path.dirname(filepath)

        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # Convert the list of dictionaries to a JSON string
        json_string = json.dumps(data, indent=4)

        # Write the JSON data to the file
        with open(filepath, 'w') as json_file:
            json_file.write(json_string)


    @classmethod
    def args2jsonEntry(cls, muX, muY, sigma, r2, signal):
        """Build the one estimate record both writers (JSON and HDF5) unpack.

        Despite the name this is NOT a JSON-specific structure: ResultFileWriter.write_h5() unpacks
        exactly these dicts. Values are kept at full precision here and rounded only by the JSON
        writer, which is a human-readable dump. Rounding used to happen right here, so every HDF5
        result was quantised to 1e-4 -- far coarser than the float32 the datasets hold.

        NOTE: sigma is stored as a magnitude. The Gaussian is even in sigma -- it enters as
        exp(-r^2 / 2*sigma^2) -- so +s and -s describe the identical pRF and nothing constrains the
        refined fit to the positive branch; it lands on whichever side the quadratic surrogate puts
        it. The sign is therefore meaningless, but it used to reach the results file, where a
        negative pRF size is simply wrong: anything downstream that averages or thresholds on sigma
        silently mis-handles those vertices. Measured on the 5000-vertex 3n2 test data, 10 of them
        came out negative.
        """
        json_entry = {
                    "Centerx0": float(muX),
                    "Centery0": float(muY),
                    "Theta": 0,
                    "sigmaMajor": abs(float(sigma)),
                    "sigmaMinor": 0,
                    "R2": float(r2),
                    "modelpred": signal.tolist()
                }
        return json_entry
    
    @classmethod
    def log_string(cls, message, filepath):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(filepath, "a") as logfile:
            logfile.write("[{}] {}\n".format(timestamp, message))
                    
        return    