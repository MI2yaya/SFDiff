from sfdiff.utils import (
    time_splitter,
    StateObsDataset
)
import numpy as np
from dataGeneration import SinusoidalWaves,Lorenz,DualSinusoidalWaves,LogisticMap,RandomWalk,xDIndependentSinusoidalWaves,TwoDDependentSinusoidalWaves, MassSpringChain, ChirpFunction, MixedSinandLogistic
import os
import math
import re
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent

def load_datapoint(config, dataset_id:str):
    dataset_type,dataset_name = config["dataset"].lower().split(':')
    
    if dataset_type == 'custom':
        dataset, generator = get_custom_dataset(dataset_name,
            samples=config['data_samples'],
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            dt=config['dt'],
            q=config['q'],
            r=config['r'],
            observation_dim=config['observation_dim'],
            plot=True
        )
    elif dataset_type == 'dataset':
        _, generator,config = get_stored_dataset(dataset_name,
            config=config,
            length=config['context_length']+config['prediction_length'],
            plot=True)
    else:
        print(f"Unknown dataset type: {dataset_type}")
        raise NotImplementedError
    
    storms = generator.parse()
    idx = generator.find_storm_index(dataset_id)
    seq = storms[idx]
    lat = seq[:,0]
    lon = seq[:,1]
    extra = seq[:,2:]
    components = encode_latlon(lat,lon)
    seq = np.concatenate([components,extra],axis=1)
    
    scaling = int(config['dt']**-1)
    context_length = config["context_length"] * scaling
    prediction_length = config["prediction_length"] * scaling
    max_len = context_length + prediction_length
    seq = seq[:max_len]
    
    
    
    formatted_seq = [
        {
            "state": seq.copy(),       # shape (seq_len, D)
            "observation": seq.copy(), # same as state
        }
    ]

    
    time_data = time_splitter(formatted_seq, context_length, prediction_length)

    processedDataset = StateObsDataset(time_data)
    print(processedDataset)
    return processedDataset, generator

def get_custom_dataset(dataset_name, samples=10, context_length=80,prediction_length=20, dt=1,q=1,r=1,observation_dim=1,plot=False):
    generatingClasses = {
        "sinusoidal": SinusoidalWaves,
        "lorenz":Lorenz,
        "dualsinusoidal": DualSinusoidalWaves,
        "logistic":LogisticMap,
        "random":RandomWalk,
        "2dsinindependent":xDIndependentSinusoidalWaves,
        "xdsinindependent":xDIndependentSinusoidalWaves,
        "2dsindependent": TwoDDependentSinusoidalWaves,
        "massspringchain":MassSpringChain,
        "chirp":ChirpFunction,
        "mixedsinandlogistic":MixedSinandLogistic,

    }
    generator = generatingClasses[dataset_name](context_length+prediction_length,dt,q,r,observation_dim)

    states = []
    observations = []
    for sample in range(samples):
        state, obs = generator.generate()
        states.append(state)
        observations.append(obs)

    state_array = np.array(states)
    observation_array = np.array(observations)

    print(state_array.shape,observation_array.shape)

    custom_data = [
        {
            "state": np.array(state),         # shape (seq_len, 1)
            "observation": np.array(obs),    # shape (seq_len, 1)
        }
        for obs, state in zip(observation_array, state_array)
    ]
    
    
    
    custom_data = np.array(custom_data)

    return custom_data, generator

def encode_latlon(lat, lon):
    # degrees → radians
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0

    return np.stack(
        [
            np.sin(lat),
            np.cos(lat),
            np.sin(lon),
            np.cos(lon),
        ],
        axis=-1,  # [..., 4]
    )

def get_stored_dataset(dataset_name, config=None,length=5,plot=False):
    retrievers = {
        "hurricane": HurdatAT, #legacy name for atlantic
        "hurricaneatlantic": HurdatAT,
        "hurricanepacific": HurdatPA,
        "hurricaneall": HurdatALL,
    }

    retriever = retrievers[dataset_name](length=length,plot=plot,observation_dim=config['observation_dim'] if config is not None and 'observation_dim' in config else 3)
    
    observations = retriever.generate()

    if config['observation_dim'] == 1: #config['use_features']:
        custom_data = []
        for obs in observations:
            obs = np.array(obs)  # [L, 3]

            lat = obs[:, 0]
            lon = obs[:, 1]
            
            features = encode_latlon(lat, lon)  # shape [L, 4]
            state = obs[:, 2:]      # windspeed, shape [L,1]

            custom_data.append({
                "features": features.astype(np.float32),
                "state": state.astype(np.float32),
                "observation": state.astype(np.float32),  # same as state for now
            })

        custom_data = np.array(custom_data)

    elif config['observation_dim'] == 4: #long lat only
        custom_data = []
        for obs in observations:
            obs = np.array(obs)  # [L, 3]

            lat = obs[:, 0]
            lon = obs[:, 1]
            
            positions = encode_latlon(lat, lon)  # shape [L, 4]
            state = positions

            custom_data.append({
                "state": state.astype(np.float32),
                "observation": state.astype(np.float32),  # same as state for now
            })

        custom_data = np.array(custom_data)

    elif config['observation_dim'] == 5: #long lat + windspeed
        custom_data = []
        for obs in observations:
            obs = np.array(obs)  # [L, 3]

            lat = obs[:, 0]
            lon = obs[:, 1]
            
            positions = encode_latlon(lat, lon)  # shape [L, 4]
            windSpeed = obs[:, 2:]      # windspeed, shape [L,1]
            state = np.concatenate([positions,windSpeed],axis=-1) #shape [L,5]

            custom_data.append({
                "state": state.astype(np.float32),
                "observation": state.astype(np.float32),  # same as state for now
            })

        custom_data = np.array(custom_data)

    else:
            
        custom_data = [
            {
                "state":np.array(obs),#only needed for batching ig,
                "observation":np.array(obs),
            }
            for obs in observations
        ]
        custom_data = np.array(custom_data)
    
    config['dt'] = retriever.dt
    config['r']=retriever.r
    config['q']=retriever.q
    config['observation_dim']=retriever.obs_dim
    config['data_samples']=len(observations)

    

    return custom_data,retriever,config


class HurdatAT:
    def __init__(self,length,plot,observation_dim=3):
        self.length=length
        self.dt= 1
        self.q = 1
        self.r = 1
        self.obs_dim = observation_dim #4 for pressure, beware NAN torture
        self.filePath = os.path.join(BASE_DIR, 'data', 'hurdat2-1851-2024-040425.txt') #Atlantic
        self.plot=plot
    
    def h_fn(self,x):
        return x
    
    def R_inv(self,resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    def _is_header_line(self, line: str) -> bool:
        """Return True if the line looks like a storm header (starts with ALxxxx)."""
        if not line:
            return False
        # Header lines usually start with 'AL' followed by digits and a comma
        return bool(re.match(r'^\s*AL\d{6}\s*,', line, flags=re.IGNORECASE))

    def sliding_windows(self, arr, win_len, overlap=0):
        step = win_len - overlap
        seq_len = arr.shape[0]
        windows = []

        for start in range(0, seq_len, step):
            end = start + win_len
            if end <= seq_len:
                windows.append(arr[start:end])
            else:
                # pad last window
                pad = np.full((win_len, arr.shape[1]), np.nan, dtype=arr.dtype)
                chunk = arr[start:]
                pad[:chunk.shape[0]] = chunk
                windows.append(pad)
                break

        return windows

    def parse(self):
        """
        Parse HURDAT2 file.
        Returns a list of numpy arrays (seq_len, 4) for each storm (no noise added here).
        """
        storms = []
        with open(self.filePath, "r") as f:
            raw_lines = f.readlines()

        # strip and keep non-empty lines
        lines = [ln.rstrip("\n") for ln in raw_lines if ln.strip() != ""]

        i = 0
        total_lines = len(lines)
        while i < total_lines:
            line = lines[i].strip()

            # if this line isn't a header for some reason, advance to next and warn
            if not self._is_header_line(line):
                print(f"[SKIP/UNEXPECTED LINE] not a header at file line {i}: {line}")
                i += 1
                continue

            # ---- header parsing ----
            header_line = line
            # extract storm id and number of obs with regex to be robust about whitespace/trailing comma
            try:
                # storm id is the first token before comma
                storm_id = header_line.split(",")[0].strip()
                # find the last integer in header (num obs)
                m = re.search(r',\s*([A-Za-z0-9_ -]+?)\s*,\s*(\d+)\s*,?$', header_line)
                if m:
                    storm_name = m.group(1).strip()
                    num_obs = int(m.group(2))
                else:
                    # fallback: split and take last non-empty token that's digits
                    parts = [p.strip() for p in header_line.split(",") if p.strip() != ""]
                    # last part should be num_obs
                    num_obs = int(parts[-1])
                    storm_name = parts[1] if len(parts) > 1 else ""
            except Exception as e:
                print(f"[HEADER PARSE ERROR] file line {i}: {header_line!r}  -- {e}")
                i += 1
                continue

            obs_list = []
            i += 1  # move to first observation line
            obs_read = 0
            while obs_read < num_obs and i < total_lines:
                obs_raw = lines[i].strip()
                # If we hit another header unexpectedly, break
                if self._is_header_line(obs_raw):
                    print(f"[UNEXPECTED HEADER DURING OBS READ] storm {storm_id} expected {num_obs} obs but header found at line {i}: {obs_raw}")
                    break

                try:
                    cols = [c.strip() for c in obs_raw.split(",")]

                    # HURDAT2 convention: date, time, record_id, status, lat, lon, wind, pressure, ...
                    # lat -> cols[4], lon -> cols[5], wind -> cols[6], pressure -> cols[7]
                    if len(cols) < 8:
                        raise ValueError(f"not enough columns ({len(cols)})")

                    lat_str = cols[4]
                    lon_str = cols[5]
                    wind_str = cols[6]
                    pressure_str = cols[7]

                    # handle missing lat/lon (empty strings)
                    if lat_str == "" or lon_str == "":
                        raise ValueError("missing lat/lon")

                    # lat like '28.0N', lon like ' 94.8W'
                    lat_dir = lat_str[-1].upper()
                    lon_dir = lon_str[-1].upper()
                    lat_val = float(lat_str[:-1])
                    lon_val = float(lon_str[:-1])

                    lat = lat_val if lat_dir == "N" else -lat_val
                    lon = lon_val if lon_dir == "E" else -lon_val

                    # convert wind/pressure, handle -999 as NaN
                    wind = float(wind_str) if wind_str not in ("-999", "") else np.nan
                    pressure = float(pressure_str) if pressure_str not in ("-999", "") else np.nan
                    

                    #Old obs dim handling
                    if self.obs_dim == 1:
                        #obs_list.append([lat])
                        obs_list.append([lat, lon, wind]) #testing 1d with features
                    elif self.obs_dim == 2:
                        obs_list.append([lat, lon])
                    elif self.obs_dim == 3:
                        obs_list.append([lat, lon, wind])
                    elif self.obs_dim == 4:
                        #obs_list.append([lat,lon,wind,pressure]) #NaN hell, be warned
                        obs_list.append([lat,lon]) #working with 4d as long lat only
                    elif self.obs_dim == 5:
                        obs_list.append([lat,lon,wind]) #splitting lat, lon, into cos and sin components
                    else:
                        raise ValueError(f"Unsupported observation dimension: {self.obs_dim}")

                    obs_read += 1

                except Exception as e:
                    print(f"[OBS PARSE ERROR] storm {storm_id} file line {i}: {obs_raw!r}  -- {e}")
                    # skip this observation but continue reading the rest
                finally:
                    i += 1

            if len(obs_list) > 0:
                storms.append(np.array(obs_list, dtype=np.float32))
            else:
                print(f"[NO VALID OBS] storm {storm_id} had 0 valid observations, skipping.")

            # continue loop (i already points to next line after the observations)
        return storms
    
    def find_storm_index(self, check_id: str) -> int:
        """
        Given a HURDAT2 file and a storm name like "KATRINA_2005",
        returns the index of that storm in the parsed list of storms (0-based).
        """
        with open(self.filePath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        storm_counter = 0
        i = 0
        while i < len(lines):
            line = lines[i]
            if self._is_header_line(line):  # header
                storm_id = line.split(",")[0].strip()
                # find the last integer in header (num obs)
                m = re.search(r',\s*([A-Za-z0-9_ -]+?)\s*,\s*(\d+)\s*,?$', line)
                if m:
                    storm_name = m.group(1).strip()
                    num_obs = int(m.group(2))
                if storm_id == check_id:
                    print(storm_name,num_obs)
                    return storm_counter
                    
                storm_counter += 1

                
            i += 1

        raise ValueError(f"Hurricane ID {check_id} not found in {self.filePath}")
  
    def find_storm_window_index(self, check_id: str) -> int:
        """
        Returns the dataset index corresponding to the FIRST window
        of the storm identified by `check_id`.
        """

        with open(self.filePath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        dataset_index = 0
        i = 0

        while i < len(lines):
            line = lines[i]

            if self._is_header_line(line):
                storm_id = line.split(",")[0].strip()

                m = re.search(r',\s*([A-Za-z0-9_ -]+?)\s*,\s*(\d+)\s*,?$', line)
                if not m:
                    raise RuntimeError(f"Failed to parse header: {line}")

                storm_name = m.group(1).strip()
                num_obs = int(m.group(2))

                # compute how many windows THIS storm produces
                win_len = self.length
                step = win_len  # no overlap in your code
                num_windows = max(1, math.ceil(num_obs / step))

                if storm_id == check_id:
                    print(f"Matched {storm_name} with {num_obs} obs → {num_windows} windows")
                    return dataset_index

                dataset_index += num_windows

                # skip obs lines
                i += num_obs + 1
            else:
                i += 1

        raise ValueError(f"Hurricane ID {check_id} not found")
    
    def generate(self):
        """Return an array shaped (seq_len, 4) with noise added."""
        storms = self.parse()
        all_windows = []
        plotted = True #set to false to plot first window only
        
        for arr in storms:
            if arr.shape[0] < 2:
                continue  


            #noise = np.random.normal(0, self.r, size=arr.shape)
            #noisy = arr + noise

            windows = self.sliding_windows(arr, self.length, 2)

            for window in windows:
                if self.plot and not plotted:
                    print(window)
                    import matplotlib.pyplot as plt
                    lat = window[:, 0]
                    lon = window[:, 1]

                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111)

                  
                    for y in range(-60, 90, 30):
                        ax.plot([-180, 180], [y, y], color="lightgray", linewidth=0.5)

                    for x in range(-180, 210, 30):
                        ax.plot([x, x], [-90, 90], color="lightgray", linewidth=0.5)

                    ax.plot(lon, lat, "-o", markersize=3)

                    ax.set_title("Storm Track (Lat/Lon)")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")

                    pad = 5
                    ax.set_xlim(np.nanmin(lon) - pad, np.nanmax(lon) + pad)
                    ax.set_ylim(np.nanmin(lat) - pad, np.nanmax(lat) + pad)

                    plt.tight_layout()
                    plt.show()
                    plotted = True  

            all_windows.extend(windows)

        return all_windows
            

class HurdatPA:
    def __init__(self,length,plot,observation_dim=3):
        self.length=length
        self.dt= 1
        self.q = 1
        self.r = 1
        self.obs_dim = observation_dim #4 for pressure, beware NAN torture
        self.filePath = os.path.join(BASE_DIR, 'data', 'hurdat2-nepac-1949-2024-031725.txt') #Pacific
        self.plot=plot
    
    def h_fn(self,x):
        return x
    
    def R_inv(self,resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    def _is_header_line(self, line: str) -> bool:
        """Return True if the line looks like a storm header (starts with ALxxxx)."""
        if not line:
            return False
        # Header lines usually start with 'AL' followed by digits and a comma
        return bool(re.match(r'^(EP\d{2}\d{4}|CP\d{2}\d{4})\s*,', line, flags=re.IGNORECASE))

    def sliding_windows(self, arr, win_len, overlap=0):
        step = win_len - overlap
        seq_len = arr.shape[0]
        windows = []

        for start in range(0, seq_len, step):
            end = start + win_len
            if end <= seq_len:
                windows.append(arr[start:end])
            else:
                # pad last window
                pad = np.full((win_len, arr.shape[1]), np.nan, dtype=arr.dtype)
                chunk = arr[start:]
                pad[:chunk.shape[0]] = chunk
                windows.append(pad)
                break

        return windows

    def parse(self):
        """
        Parse HURDAT2 file.
        Returns a list of numpy arrays (seq_len, 4) for each storm (no noise added here).
        """
        storms = []
        with open(self.filePath, "r") as f:
            raw_lines = f.readlines()

        # strip and keep non-empty lines
        lines = [ln.rstrip("\n") for ln in raw_lines if ln.strip() != ""]

        i = 0
        total_lines = len(lines)
        while i < total_lines:
            line = lines[i].strip()

            # if this line isn't a header for some reason, advance to next and warn
            if not self._is_header_line(line):
                print(f"[SKIP/UNEXPECTED LINE] not a header at file line {i}: {line}")
                i += 1
                continue

            # ---- header parsing ----
            header_line = line
            # extract storm id and number of obs with regex to be robust about whitespace/trailing comma
            try:
                # storm id is the first token before comma
                storm_id = header_line.split(",")[0].strip()
                # find the last integer in header (num obs)
                m = re.search(r',\s*([A-Za-z0-9_ -]+?)\s*,\s*(\d+)\s*,?$', header_line)
                if m:
                    storm_name = m.group(1).strip()
                    num_obs = int(m.group(2))
                else:
                    # fallback: split and take last non-empty token that's digits
                    parts = [p.strip() for p in header_line.split(",") if p.strip() != ""]
                    # last part should be num_obs
                    num_obs = int(parts[-1])
                    storm_name = parts[1] if len(parts) > 1 else ""
            except Exception as e:
                print(f"[HEADER PARSE ERROR] file line {i}: {header_line!r}  -- {e}")
                i += 1
                continue

            obs_list = []
            i += 1  # move to first observation line
            obs_read = 0
            while obs_read < num_obs and i < total_lines:
                obs_raw = lines[i].strip()
                # If we hit another header unexpectedly, break
                if self._is_header_line(obs_raw):
                    print(f"[UNEXPECTED HEADER DURING OBS READ] storm {storm_id} expected {num_obs} obs but header found at line {i}: {obs_raw}")
                    break

                try:
                    cols = [c.strip() for c in obs_raw.split(",")]

                    # HURDAT2 convention: date, time, record_id, status, lat, lon, wind, pressure, ...
                    # lat -> cols[4], lon -> cols[5], wind -> cols[6], pressure -> cols[7]
                    if len(cols) < 8:
                        raise ValueError(f"not enough columns ({len(cols)})")

                    lat_str = cols[4]
                    lon_str = cols[5]
                    wind_str = cols[6]
                    pressure_str = cols[7]

                    # handle missing lat/lon (empty strings)
                    if lat_str == "" or lon_str == "":
                        raise ValueError("missing lat/lon")

                    # lat like '28.0N', lon like ' 94.8W'
                    lat_dir = lat_str[-1].upper()
                    lon_dir = lon_str[-1].upper()
                    lat_val = float(lat_str[:-1])
                    lon_val = float(lon_str[:-1])

                    lat = lat_val if lat_dir == "N" else -lat_val
                    lon = lon_val if lon_dir == "E" else -lon_val

                    # convert wind/pressure, handle -999 as NaN
                    wind = float(wind_str) if wind_str not in ("-999", "") else np.nan
                    pressure = float(pressure_str) if pressure_str not in ("-999", "") else np.nan
                    

                    #Old obs dim handling
                    if self.obs_dim == 1:
                        #obs_list.append([lat])
                        obs_list.append([lat, lon, wind]) #testing 1d with features
                    elif self.obs_dim == 2:
                        obs_list.append([lat, lon])
                    elif self.obs_dim == 3:
                        obs_list.append([lat, lon, wind])
                    elif self.obs_dim == 4:
                        obs_list.append([lat,lon,wind,pressure]) #NaN hell, be warned
                    else:
                        raise ValueError(f"Unsupported observation dimension: {self.obs_dim}")

                    obs_read += 1

                except Exception as e:
                    print(f"[OBS PARSE ERROR] storm {storm_id} file line {i}: {obs_raw!r}  -- {e}")
                    # skip this observation but continue reading the rest
                finally:
                    i += 1

            if len(obs_list) > 0:
                storms.append(np.array(obs_list, dtype=np.float32))
            else:
                print(f"[NO VALID OBS] storm {storm_id} had 0 valid observations, skipping.")

            # continue loop (i already points to next line after the observations)
        return storms
    
    def generate(self):
        """Return an array shaped (seq_len, 4) with noise added."""
        storms = self.parse()
        all_windows = []
        plotted = True #set to false to plot first window only
        
        for arr in storms:
            if arr.shape[0] < 2:
                continue  


            noise = np.random.normal(0, self.r, size=arr.shape)
            noisy = arr + noise

            windows = self.sliding_windows(noisy, self.length, 2)

            for window in windows:
                if self.plot and not plotted:
                    print(window)
                    import matplotlib.pyplot as plt
                    lat = window[:, 0]
                    lon = window[:, 1]

                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111)

                  
                    for y in range(-60, 90, 30):
                        ax.plot([-180, 180], [y, y], color="lightgray", linewidth=0.5)

                    for x in range(-180, 210, 30):
                        ax.plot([x, x], [-90, 90], color="lightgray", linewidth=0.5)

                    ax.plot(lon, lat, "-o", markersize=3)

                    ax.set_title("Storm Track (Lat/Lon)")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")

                    pad = 5
                    ax.set_xlim(np.nanmin(lon) - pad, np.nanmax(lon) + pad)
                    ax.set_ylim(np.nanmin(lat) - pad, np.nanmax(lat) + pad)

                    plt.tight_layout()
                    plt.show()
                    plotted = True  

            all_windows.extend(windows)

        return all_windows
            
            
class HurdatALL:
    def __init__(self,length,plot,observation_dim=3):
        self.length=length
        self.dt= 1
        self.q = 1
        self.r = 1
        self.obs_dim = observation_dim #4 for pressure, beware NAN torture
        self.atlanticRetriever = HurdatAT(length,plot,observation_dim)
        self.pacificRetriever = HurdatPA(length,plot,observation_dim)
        self.plot=plot
    
    def h_fn(self,x):
        return x
    
    def R_inv(self,resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    def generate(self):
        """Return an array shaped (seq_len, 4) with noise added."""
        atlantic_windows = self.atlanticRetriever.generate()
        pacific_windows = self.pacificRetriever.generate()
        
        all_windows = atlantic_windows + pacific_windows
        return all_windows