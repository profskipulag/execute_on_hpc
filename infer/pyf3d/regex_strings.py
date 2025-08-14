
# Here we list all the different sorts of things we want to detect in a .inp file ...

# ... detects any float of the form 1.0, -1.0, +1.0, -1., +1., etc.
any_float = "[-+]?(?:\d*\.*\d+)[.]?"

# ... detects any integer of the form +1, -1, 1, 999, etc.
any_int = "[-+]?[0-9]+"

# ... detects any text without whitespace
any_non_ws = "\S*"

# ... detects any sequence of floats or integers separated  by whitespace
any_sequence_of_floats_and_integers = "([+-\.\d\s]*[\.\d])" #"([+-\.\d\s]+)\s"

# ... detects any of these variations on yes and no - add more when you come across them
any_yesno = "YES|Yes|yes|NO|No|no"

# ... detects any of these variations of on or off -  add more when you come across them
any_onoff = "ON|On|on|OFF|Off|off"

# ... different sorts of perturbation - this is repeated a lot in the perturbation section, so I include it here
any_perturbation = "NO|RELATIVE|ABSOLUTE"

# ... different sorts of perturbation PDF - this is repeated a lot in the perturbation section, so I include it here
any_perturbation_pdf = "UNIFORM|GAUSSIAN"

# ... we put them all inn a dict so that we can insert them into regex strings ...

regexs = {
    'any_float':any_float,
    'any_int':any_int,
    'any_non_ws':any_non_ws,
    'any_yesno':any_yesno,
    'any_onoff':any_onoff,
    'any_perturbation':any_perturbation,
    'any_perturbation_pdf':any_perturbation_pdf,
    'any_sequence_of_floats_and_integers':any_sequence_of_floats_and_integers
    
}

# ... and here are the rgeex strings, one for each section of a .inp file.
# The search terms described above are inserted by:  regex_string.format(**regexs)
# One of search terms aren't listed in the dict above, but are written into the 
# strings below:

regex_time_utc = """
.*
YEAR  = (?P<year>{any_int}).*
MONTH = (?P<month>{any_int}).*
DAY   = (?P<day>{any_int}).*
RUN_START_\(HOURS_AFTER_00\) = (?P<run_start>{any_int}).*
RUN_END_\(HOURS_AFTER_00\)   = (?P<run_end>{any_int}).*
INITIAL_CONDITION          = (?P<initial_condition>NONE|RESTART|INSERTION).*
RESTART_FILE               = (?P<restart_file>{any_non_ws}).*
RESTART_ENSEMBLE_BASEPATH  = (?P<restart_ensemble_basepath>{any_non_ws}).*
""".format(**regexs).replace("\n","")


regex_insertion_data = """
.*
INSERTION_FILE             = (?P<insertion_file>{any_non_ws}).*
INSERTION_DICTIONARY_FILE  = (?P<insertion_dictionary_file>{any_non_ws}).*
INSERTION_TIME_SLAB        = (?P<insertion_time_slab>{any_int}).*
DIAMETER_CUT_OFF_\(MIC\)     = (?P<diameter_cut_off>{any_int}).*
.*""".format(**regexs).replace("\n","")

regex_meteo = """ 
.*
METEO_DATA_FORMAT          = (?P<meteo_data_format>{any_non_ws}).*
METEO_DATA_DICTIONARY_FILE = (?P<meteo_data_dictionary_file>{any_non_ws}).*
METEO_DATA_FILE            = (?P<meteo_data_file>{any_non_ws}).*
METEO_ENSEMBLE_BASEPATH    = (?P<meteo_ensemble_basepath>{any_non_ws}).*
METEO_LEVELS_FILE          = (?P<meteo_levels_file>{any_non_ws}).*
DBS_BEGIN_METEO_DATA_\(HOURS_AFTER_00\) = (?P<dbs_begin_meteo_data>{any_int}).*
DBS_END_METEO_DATA_\(HOURS_AFTER_00\)   = (?P<dbs_end_meteo_data>{any_int}).*
METEO_COUPLING_INTERVAL_\(MIN\)         = (?P<meteo_coupling_interval>{any_int}).*
MEMORY_CHUNK_SIZE = (?P<memory_chunk_size>{any_int}).*
""".format(**regexs).replace("\n","")

regex_grid = """ 
.*
HORIZONTAL_MAPPING = (?P<horizontal_mapping>{any_non_ws}).*
VERTICAL_MAPPING   = (?P<vertical_mapping>{any_non_ws}).*
LONMIN = (?P<lonmin>{any_float}).*
LONMAX = (?P<lonmax>{any_float}).*
LATMIN = (?P<latmin>{any_float}).*
LATMAX = (?P<latmax>{any_float}).*
NX = (?P<nx>{any_int}).*
NY = (?P<ny>{any_int}).*
NZ = (?P<nz>{any_int}).*
ZMAX_\(M\) = (?P<zmax>{any_int}).*
""".format(**regexs).replace("\n","")

regex_species = """ 
.*
TEPHRA = (?P<tephra>{any_onoff}).*
DUST   = (?P<dust>{any_onoff}).*
H2O    = (?P<h2o>{any_onoff})   MASS_FRACTION_\(\%\) = (?P<h2o_mass_fraction>{any_float}).*
SO2    = (?P<so2>{any_onoff})    MASS_FRACTION_\(\%\) = (?P<so2_mass_fraction>{any_float}).*
CS134  = (?P<cs134>{any_onoff})   MASS_FRACTION_\(\%\) = (?P<cs134_mass_fraction>{any_float}).*
CS137  = (?P<cs137>{any_onoff})   MASS_FRACTION_\(\%\) = (?P<cs137_mass_fraction>{any_float}).*
I131   = (?P<i131>{any_onoff})   MASS_FRACTION_\(\%\) = (?P<i131_mass_fraction>{any_float}).*
SR90   = (?P<sr90>{any_onoff})   MASS_FRACTION_\(\%\) = (?P<sr90_mass_fraction>{any_float}).*
Y90    = (?P<y90>{any_onoff})   MASS_FRACTION_\(\%\) = (?P<y90_mass_fraction>{any_float}).*
""".format(**regexs).replace("\n","")

regex_tephra_tgsd = """ 
.*
NUMBER_OF_BINS   = (?P<number_of_bins>{any_int}).*
FI_RANGE         = (?P<fi_range1>{any_int})\s+(?P<fi_range2>{any_int}).*
DENSITY_RANGE    = (?P<density_range1>{any_int})\s+(?P<density_range2>{any_int}).*
SPHERICITY_RANGE = (?P<sphericity_range1>{any_float})\s+(?P<sphericity_range2>{any_float}).*
DISTRIBUTION     =  (?P<distribution>GAUSSIAN|BIGAUSSIAN|WEIBULL|BIWEIBULL|CUSTOM|ESTIMATE).*
IF_GAUSSIAN    FI_MEAN  = (?P<gaussian_fi_mean>{any_float})\s+FI_DISP = (?P<gaussian_fi_disp>{any_float}).*
IF_BIGAUSSIAN  FI_MEAN  = (?P<bigaussian_fi_mean1>{any_float})\s+(?P<bigaussian_fi_mean2>{any_float})\s+FI_DISP = (?P<bigaussian_fi_disp1>{any_float})\s+(?P<bigaussian_fi_disp2>{any_float})\s+MIXING_FACTOR = (?P<bigaussian_mixing_factor>{any_float}).*
IF_WEIBULL     FI_SCALE = (?P<weibull_fi_scale>{any_float})\s+W_SHAPE = (?P<weibull_w_shape>{any_float})\s+
IF_BIWEIBULL   FI_SCALE = (?P<biweibull_fi_scale1>{any_float})\s+(?P<biweibull_fi_scale2>{any_float})\s+W_SHAPE =\s+(?P<biweibull_w_shape1>{any_float})\s+(?P<biweibull_w_shape2>{any_float})\s+MIXING_FACTOR =\s+(?P<biweibull_mixing_factor>{any_float}).*
IF_CUSTOM      FILE = (?P<custom_file>{any_non_ws}).*
IF_ESTIMATE    VISCOSITY_\(PAS\) = (?P<estimate_viscosity>{any_non_ws})\s+HEIGHT_ABOVE_VENT_\(M\) = (?P<estimate_height_above_vent>{any_float})
.*
""".format(**regexs).replace("\n","")

regex_radionucleides_tgsd = """ 
.*
NUMBER_OF_BINS   = (?P<number_of_bins>{any_int}).*
FI_RANGE         = (?P<fi_range1>{any_int})\s+(?P<fi_range2>{any_int}).*
DENSITY_RANGE    = (?P<density_range1>{any_int})\s+(?P<density_range2>{any_int}).*
SPHERICITY_RANGE = (?P<sphericity_range1>{any_float})\s+(?P<sphericity_range2>{any_float}).*
DISTRIBUTION     = (?P<distribution>GAUSSIAN|BIGAUSSIAN|WEIBULL|BIWEIBULL|CUSTOM|ESTIMATE).*
IF_GAUSSIAN    FI_MEAN  = (?P<gaussian_fi_mean>{any_float})\s+FI_DISP = (?P<gaussian_fi_disp>{any_float}).*
IF_BIGAUSSIAN  FI_MEAN  = (?P<bigaussian_fi_mean1>{any_float})\s+(?P<bigaussian_fi_mean2>{any_float})\s+FI_DISP = (?P<bigaussian_fi_disp1>{any_float})\s+(?P<bigaussian_fi_disp2>{any_float})\s+MIXING_FACTOR = (?P<bigaussian_mixing_factor>{any_float}).*
IF_WEIBULL     FI_SCALE = (?P<weibull_fi_scale>{any_float})\s+W_SHAPE = (?P<weibull_w_shape>{any_float})\s+
IF_BIWEIBULL   FI_SCALE = (?P<biweibull_fi_scale1>{any_float})\s+(?P<biweibull_fi_scale2>{any_float})\s+W_SHAPE =\s+(?P<biweibull_w_shape1>{any_float})\s+(?P<biweibull_w_shape2>{any_float})\s+MIXING_FACTOR =\s+(?P<biweibull_mixing_factor>{any_float}).*
IF_CUSTOM      FILE = (?P<custom_file>{any_non_ws}).*
IF_ESTIMATE    VISCOSITY_\(PAS\) = (?P<estimate_viscosity>{any_non_ws})\s+HEIGHT_ABOVE_VENT_\(M\) = (?P<estimate_height_above_vent>{any_float})
.*
""".format(**regexs).replace("\n","")

regex_particle_aggregation = """ 
.*
PARTICLE_CUT_OFF          = (?P<particle_cut_off>NONE|FI_LARGER_THAN|FI_LOWER_THAN).*
AGGREGATION_MODEL         = (?P<aggregation_model>NONE|PERCENTAGE|CORNELL|COSTA).*
NUMBER_OF_AGGREGATE_BINS  = (?P<number_of_aggregate_bins>{any_int}).*
DIAMETER_AGGREGATES_\(MIC\) = (?P<diameter_aggregates1>{any_float})\s+(?P<diameter_aggregates2>{any_float}).*
DENSITY_AGGREGATES_\(KGM3\) = (?P<density_aggregates1>{any_float})\s+(?P<density_aggregates2>{any_float}).*
PERCENTAGE_\(\%\)            = (?P<percentage1>{any_float})\s+(?P<percentage2>{any_float}).*
VSET_FACTOR               = (?P<vset_factor>{any_float}).*
FRACTAL_EXPONENT          = (?P<fractal_exponent>{any_float}).*
.*
""".format(**regexs).replace("\n","")

regex_source = """ 
.*
SOURCE_TYPE                   = (?P<source_type>POINT|SUZUKI|TOP-HAT|PLUME|RESUSPENSION).*
SOURCE_START_\(HOURS_AFTER_00\) = (?P<source_start>{any_sequence_of_floats_and_integers}).*
SOURCE_END_\(HOURS_AFTER_00\)   = (?P<source_end>{any_sequence_of_floats_and_integers}).*
LON_VENT        = (?P<lon_vent>{any_float}).*
LAT_VENT        = (?P<lat_vent>{any_float}).*
VENT_HEIGHT_\(M\) = (?P<vent_height>{any_float}).*
HEIGHT_ABOVE_VENT_\(M\)         = (?P<height_above_vent>{any_sequence_of_floats_and_integers}).*
MASS_FLOW_RATE_\(KGS\)          = (?P<mass_flow_rate>{any_sequence_of_floats_and_integers}).*
ALFA_PLUME                    = (?P<alfa_plume>{any_float}).*
BETA_PLUME                    = (?P<beta_plume>{any_float}).*
EXIT_TEMPERATURE_\(K\)          = (?P<exit_temperature>{any_float}).*
EXIT_WATER_FRACTION_\(\%\)       = (?P<exit_water_fraction>{any_float}).*
A = (?P<a1>{any_float})\s+(?P<a2>{any_float}).*
L = (?P<l>{any_float}).*
THICKNESS_\(M\) = (?P<thickness>{any_float}).*
SOLVE_PLUME_FOR                   = (?P<solve_plume_for>MFR|HEIGHT).*
MFR_SEARCH_RANGE                  = (?P<mfr_search_range1>{any_int})\s+(?P<mfr_search_range2>{any_int}).*
EXIT_VELOCITY_\(MS\)                = (?P<exit_velocity>{any_float}).*
EXIT_GAS_WATER_TEMPERATURE_\(K\)    = (?P<exit_gas_water_temperature>{any_float}).*
EXIT_LIQUID_WATER_TEMPERATURE_\(K\) = (?P<exit_liquid_water_temperature>{any_float}).*
EXIT_SOLID_WATER_TEMPERATURE_\(K\)  = (?P<exit_solid_water_temperature>{any_float}).*
EXIT_GAS_WATER_FRACTION_\(\%\)       = (?P<exit_gas_water_fraction>{any_float}).*
EXIT_LIQUID_WATER_FRACTION_\(\%\)    = (?P<exit_liquid_water_fraction>{any_float}).*
EXIT_SOLID_WATER_FRACTION_\(\%\)     = (?P<exit_solid_water_fraction>{any_float}).*
WIND_COUPLING = (?P<wind_coupling>{any_yesno}).*
AIR_MOISTURE  = (?P<air_moisture>{any_yesno}).*
LATENT_HEAT   = (?P<latent_heat>{any_yesno}).*
REENTRAINMENT = (?P<reentrainment>{any_yesno}).*
BURSIK_FACTOR = (?P<bursik_factor>{any_float}).*
Z_MIN_WIND    = (?P<z_min_wind>{any_float}).*
C_UMBRELLA    = (?P<c_umbrella>{any_float}).*
A_S           = (?P<a_s>CONSTANT\s+{any_float}\s+{any_float}|KAMISNKY-R|KAMINSKY-C|OLD).*
A_V           = (?P<a_v>CONSTANT\s+{any_float}|TATE).*
.*
""".format(**regexs).replace("\n","")

regex_ensemble = """ 
.*
RANDOM_NUMBERS_FROM_FILE  = (?P<random_numbers_from_file>{any_yesno}).*
PERTURBATE_COLUMN_HEIGHT  = (?P<perturbate_column_height>{any_perturbation}).*
PERTURBATION_RANGE = (?P<column_height_perturbation_range>{any_int}).*
PDF                = (?P<column_height_pdf>{any_perturbation_pdf}).*
PERTURBATE_MASS_FLOW_RATE = (?P<perturbate_mass_flow_rate>{any_perturbation}).*
PERTURBATION_RANGE = (?P<mass_flow_rate_perturbation_range>{any_int}).*
PDF                = (?P<mass_flow_rate_pdf>{any_perturbation_pdf}).*
PERTURBATE_SOURCE_START = (?P<perturbate_source_start>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_source_start_range>{any_float}).*
PDF                = (?P<perturbate_source_start_pdf>{any_perturbation_pdf}).*
PERTURBATE_SOURCE_DURATION = (?P<perturbate_source_duration>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_source_duration_range>{any_float}).*
PDF                = (?P<perturbate_source_duration_pdf>{any_perturbation_pdf}).*
PERTURBATE_TOP-HAT_THICKNESS = (?P<perturbate_top_hat_thickness>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_top_hat_thickness_range>{any_float}).*
PDF                = (?P<perturbate_top_hat_pdf>{any_perturbation_pdf}).*
PERTURBATE_SUZUKI_A = (?P<perturbate_suzuki_a>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_suzuki_a_range>{any_float}).*
PDF                = (?P<perturbate_suzuki_a_pdf>{any_perturbation_pdf}).*
PERTURBATE_SUZUKI_L = (?P<perturbate_suzuki_l>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_suzuki_l_range>{any_float}).*
PDF                = (?P<perturbate_suzuki_l_pdf>{any_perturbation_pdf}).*
PERTURBATE_WIND = (?P<perturbate_wind>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_wind_range>{any_float}).*
PDF                = (?P<perturbate_wind_pdf>{any_perturbation_pdf}).*
PERTURBATE_DATA_INSERTION_CLOUD_HEIGHT  = (?P<perturbate_data_insertion_cloud_height>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_data_insertion_cloud_height_range>{any_float}).*
PDF                = (?P<perturbate_data_insertion_cloud_height_pdf>{any_perturbation_pdf}).*
PERTURBATE_DATA_INSERTION_CLOUD_THICKNESS = (?P<perturbate_data_insertion_cloud_thickness>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_data_insertion_cloud_thickness_range>{any_float}).*
PDF                = (?P<perturbate_data_insertion_cloud_thickness_pdf>{any_perturbation_pdf}).*
PERTURBATE_FI_MEAN = (?P<perturbate_fi_mean>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_fi_range>{any_float}).*
PDF                = (?P<perturbate_fi_pdf>{any_perturbation_pdf}).*
PERTURBATE_DIAMETER_AGGREGATES_\(MIC\) = (?P<perturbate_diamater_aggregates>{any_yesno}).*
PERTURBATION_RANGE = (?P<perturbate_diamater_aggregates_range>{any_float}).*
PDF                = (?P<perturbate_diamater_aggregates_pdf>{any_perturbation_pdf}).*
PERTURBATE_DENSITY_AGGREGATES = (?P<perturbate_density_aggregates>{any_perturbation}).*
PERTURBATION_RANGE = (?P<perturbate_density_aggregates_range>{any_float}).*
PDF                = (?P<perturbate_density_aggregates_pdf>{any_perturbation_pdf}).*
.*
""".format(**regexs).replace("\n","")


regex_ensemble_postprocess = """ 
.*
POSTPROCESS_MEMBERS      = (?P<postprocess_members>{any_yesno}).*
POSTPROCESS_MEAN         = (?P<postprocess_mean>{any_yesno}).*
POSTPROCESS_LOGMEAN      = (?P<postprocess_logmean>{any_yesno}).*
POSTPROCESS_MEDIAN       = (?P<postprocess_median>{any_yesno}).*
POSTPROCESS_STANDARD_DEV = (?P<postprocess_standard_dev>{any_yesno}).*
POSTPROCESS_PROBABILITY  = (?P<postprocess_probability>{any_yesno}).*
POSTPROCESS_PERCENTILES  = (?P<postprocess_percentiles>{any_yesno}).*
CONCENTRATION_THRESHOLDS_\(MG\/M3\) = (?P<postprocess_probability_concentration_thresholds>{any_float}).*
COLUMN_MASS_THRESHOLDS_\(G\/M2\)    = (?P<postprocess_probability_column_mass_thresholds_gm2>{any_float}).*
COLUMN_MASS_THRESHOLDS_\(DU\)      = (?P<postprocess_probability_column_mass_thresholds_du>{any_float}).*
GROUND_LOAD_THRESHOLDS_\(KG\/M2\)   = (?P<postprocess_probability_ground_load_thresholds>{any_float}).* 
PERCENTILE_VALUES_\(\%\) = (?P<postprocess_percentiles_percentile_values>{any_float}).*  
.*
""".format(**regexs).replace("\n","")

regex_model_physics = """ 
.*
LIMITER                     = (?P<limiter>MINMOD|SUPERBEE|OSPRE).*
TIME_MARCHING               = (?P<time_marching>EULER|RUNGE-KUTTA).*
CFL_CRITERION               = (?P<cfl_criterion>ONE_DIMENSIONAL|ALL_DIMENSIONS).*
CFL_SAFETY_FACTOR           = (?P<cfl_safety_factor>{any_float}).*
TERMINAL_VELOCITY_MODEL     = (?P<terminal_velocity_model>ARASTOOPOUR|GANSER|WILSON|DELLINO|PFEIFFER|DIOGUARDI2017|DIOGUARDI2018).*
HORIZONTAL_TURBULENCE_MODEL = (?P<horizontal_turbulence_model>CONSTANT = {any_float}|CMAQ|RAMS).*
VERTICAL_TURBULENCE_MODEL   constant = (?P<vertical_turbulence_model>{any_float}).*
RAMS_CS                     = (?P<rams_cs>{any_float}).*
WET_DEPOSITION              = (?P<wet_deposition>{any_yesno}).*
DRY_DEPOSITION              = (?P<dry_deposition>{any_yesno}).*
GRAVITY_CURRENT             = (?P<gravity_current>{any_yesno}).*
C_FLOW_RATE               = (?P<c_flow_rate>{any_non_ws}).*
LAMBDA_GRAV               = (?P<lambda_grav>{any_float}).*
K_ENTRAIN                 = (?P<k_entrain>{any_float}).*
BRUNT_VAISALA             = (?P<brunt_vaisala>{any_float}).*
GC_START_\(HOURS_AFTER_00\) = (?P<gc_start>{any_int}).*
GC_END_\(HOURS_AFTER_00\)   = (?P<gc_end>{any_int}).*
.*
""".format(**regexs).replace("\n","")

regex_model_output = """ 
.*
PARALLEL_IO                   = (?P<parallel_io>{any_yesno}).*
LOG_FILE_LEVEL                = (?P<log_file_level>NONE|NORMAL|FULL).*
RESTART_TIME_INTERVAL_\(HOURS\) = (?P<restart_time_interval>NONE|END_ONLY|{any_float}).*
OUTPUT_INTERMEDIATE_FILES     = (?P<output_intermediate_files>{any_yesno}).*
OUTPUT_TIME_START_\(HOURS\)     = (?P<output_time_start>RUN_START|{any_int}).*
OUTPUT_TIME_INTERVAL_\(HOURS\)  = (?P<output_time_interval>{any_int}).*
OUTPUT_3D_CONCENTRATION       = (?P<output_3d_concentration>{any_yesno}).*
OUTPUT_3D_CONCENTRATION_BINS  = (?P<output_3d_concentration_bins>{any_yesno}).*
OUTPUT_SURFACE_CONCENTRATION  = (?P<output_surface_concentration>{any_yesno}).*
OUTPUT_COLUMN_LOAD            = (?P<output_column_load>{any_yesno}).*
OUTPUT_CLOUD_TOP              = (?P<output_cloud_top>{any_yesno}).*
OUTPUT_GROUND_LOAD            = (?P<output_ground_load>{any_yesno}).*
OUTPUT_GROUND_LOAD_BINS       = (?P<output_ground_load_bins>{any_yesno}).*
OUTPUT_WET_DEPOSITION         = (?P<output_wet_deposition>{any_yesno}).*
TRACK_POINTS                  = (?P<output_track_points>{any_yesno}).*
TRACK_POINTS_FILE             = (?P<output_track_points_file>{any_non_ws}).*
OUTPUT_CONCENTRATION_AT_XCUTS = (?P<output_concentrations_at_xcuts>{any_yesno}).*
X-VALUES  (?P<x_values>{any_non_ws}).*
OUTPUT_CONCENTRATION_AT_YCUTS = (?P<output_concentrations_at_ycuts>{any_yesno}).*
Y-VALUES  (?P<y_values>{any_non_ws}).*
OUTPUT_CONCENTRATION_AT_ZCUTS = (?P<output_concentrations_at_zcuts>{any_yesno}).*
Z-VALUES  (?P<z_values>{any_non_ws}).*
OUTPUT_CONCENTRATION_AT_FL    = (?P<output_concentrations_at_fl>{any_yesno}).*
FL-VALUES  (?P<fl_values>[0-9 ]+).*
.*
""".format(**regexs).replace("\n","")

regex_model_validation = """ 
.*
OBSERVATIONS_TYPE            = (?P<observations_type>SATELLITE_DETECTION|SATELLITE_RETRIEVAL|DEPOSIT_CONTOURS|DEPOSIT_POINTS).*
OBSERVATIONS_FILE            = (?P<observations_file>{any_non_ws}).*
OBSERVATIONS_DICTIONARY_FILE = (?P<observations_dictionary_file>{any_non_ws}).*
RESULTS_FILE                 = (?P<results_file>{any_non_ws}).*
COLUMN_MASS_OBSERVATION_THRESHOLD_\(G/M2\)  = (?P<column_mass_observations_threshold_gm2>{any_float}).*
COLUMN_MASS_OBSERVATION_THRESHOLD_\(DU\)    = (?P<column_mass_observations_threshold_du>{any_float}).*
GROUND_LOAD_OBSERVATION_THRESHOLD_\(KG\/M2\) = (?P<ground_load_observation_threshold_kgm2>{any_float}).* 
.*
""".format(**regexs).replace("\n","")
