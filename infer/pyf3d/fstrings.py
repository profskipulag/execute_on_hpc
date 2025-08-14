

fstring_boilerplate = """!  
!-----------------------------------------------------
!
!   FALL3D : EXAMPLE INPUT FILE
!   VERSION: 8.x
!   NOTE   : This file has no backwards compatibility
!            with versions 7.x or lower
!
!-----------------------------------------------------
!
 ----
 CODE
 ----
   !
   VERSION 8.3
   !
"""


fstring_time_utc = """ --------
 TIME_UTC
 --------
   !
   !    YEAR                                : value (YYYY)
   !    MONTH                               : value (MM)
   !    DAY                                 : value (DD)
   !    RUN_START_(HOURS_AFTER_00)          : value
   !    RUN_END_(HOURS_AFTER_00)            : value
   !    INITIAL_CONDITION           options : NONE / RESTART / INSERTION
   !    RESTART_FILE                        : path to the restart           file (only used if INITIAL_CONDITION = RESTART  )
   !    RESTART_ENSEMBLE_BASEPATH           : Root path for RESTART_FILE. Used for ensemble runs when multiple restart files are available ($RESTART_ENSEMBLE_BASEPATH/0001/$RESTART_FILE...). If not provided a single restart file is used for the ensemble ($RESTART_FILE)
   !
   YEAR  = {year:04}
   MONTH = {month:02}
   DAY   = {day:02}
   RUN_START_(HOURS_AFTER_00) = {run_start}
   RUN_END_(HOURS_AFTER_00)   = {run_end}
   INITIAL_CONDITION          = {initial_condition}
   RESTART_FILE               = {restart_file}
   RESTART_ENSEMBLE_BASEPATH  = {restart_ensemble_basepath}
   !
"""


fstring_insertion_data =""" --------------
 INSERTION_DATA
 --------------
   !
   !    INSERTION_FILE                    : path to the initial condition file (only used if INITIAL_CONDITION = INSERTION)
   !    INSERTION_DICTIONARY_FILE         : Optional. Path to the insertion dictionary file. If not given, dafault value is used.
   !    INSERTION_TIME_SLAB               : value. Time slab (in the netCDF file) at which data is inserted
   !    DIAMETER_CUT_OFF_(MIC)            : value, optional. If given, assimilate only for particles smaller than cut-off value
   !
   INSERTION_FILE             = {insertion_file}
   INSERTION_DICTIONARY_FILE  = {insertion_dictionary_file}
   INSERTION_TIME_SLAB        = {insertion_time_slab}
   DIAMETER_CUT_OFF_(MIC)     = {diameter_cut_off}
   !
"""


fstring_meteo =""" ----------
 METEO_DATA
 ----------
   !
   !    METEO_DATA_FORMAT             options : WRF/ GFS / ERA5 / ERA5ML / IFS / CARRA
   !    METEO_DATA_DICTIONARY_FILE            : Optional. Path to the meteo model dictionary file. If not given, dafault value is used.
   !    METEO_DATA_FILE                       : path to the meteo model data file
   !    METEO_ENSEMBLE_BASEPATH               : Root path for METEO_DATA_FILE. Used for ensemble runs when multiple meteo files are available ($METEO_ENSEMBLE_BASEPATH/0001/$METEO_DATA_FILE...). If not provided a single meteo file is used for the ensemble (METEO_DATA_FILE)
   !    METEO_LEVELS_FILE                     : path to the meteo model levels file. Only used if METEO_DATA_FORMAT = ERA5ML or IFS
   !    DBS_BEGIN_METEO_DATA_(HOURS_AFTER_00) : value.
   !    DBS_END_METEO_DATA_(HOURS_AFTER_00)   : value.
   !    METEO_COUPLING_INTERVAL_(MIN)         : value. Time interval update of meteorological variables (does not apply to velocity)
   !    MEMORY_CHUNK_SIZE                     : value. Size of memory chunks used to store meteo data timesteps. Must be greater than 1
   !
   METEO_DATA_FORMAT          = {meteo_data_format}
   METEO_DATA_DICTIONARY_FILE = {meteo_data_dictionary_file}
   METEO_DATA_FILE            = {meteo_data_file}
   METEO_ENSEMBLE_BASEPATH    = {meteo_ensemble_basepath}
   METEO_LEVELS_FILE          = {meteo_levels_file}
   !
   DBS_BEGIN_METEO_DATA_(HOURS_AFTER_00) = {dbs_begin_meteo_data}
   DBS_END_METEO_DATA_(HOURS_AFTER_00)   = {dbs_end_meteo_data}
   METEO_COUPLING_INTERVAL_(MIN)         = {meteo_coupling_interval}
   !
   MEMORY_CHUNK_SIZE = {memory_chunk_size}
   !
"""

fstring_grid = """ ----
 GRID
 ----
   !
   !    HORIZONTAL_MAPPING options  : CARTESIAN / SPHERICAL 
   !    VERTICAL_MAPPING   options  : SIGMA_NO_DECAY / SIGMA_LINEAR_DECAY / SIGMA_EXPONENTIAL_DECAY
   !    LONMIN                      : value. Longitude of the grid W side in the range (-180,180)
   !    LONMAX                      : value. Longitude of the grid E side in the range (-180,180)
   !    LATMIN                      : value. Latitude  of the grid S side in the range ( -90,90 )
   !    LATMAX                      : value. Latitude  of the grid N side in the range ( -90,90 )
   !    NX [RESOLUTION=value]       : value. Number of grid cells (mass points) along x.
   !                                  If optional argument RESOLUTION exists then the value of NX is calculated
   !    NY [RESOLUTION=value]       : value. Number of grid cells (mass points) along y
   !                                  If optional argument RESOLUTION exists then the value of NY is calculated
   !    NZ                          : value. Number of grid cells (mass points) along z
   !    ZMAX_(M)                    : value. Top of the computational domain (in m)
   !    SIGMA_VALUES (optional)     : values of sigma coordinate in the range (0,1) where sigma = X3/X3max. The number of values
   !                                  has to be less or equal than NZ+1. If not present, uniform distribution is assumed.
   !
   HORIZONTAL_MAPPING = {horizontal_mapping}
   VERTICAL_MAPPING   = {vertical_mapping}
   LONMIN = {lonmin}
   LONMAX = {lonmax}
   LATMIN = {latmin}
   LATMAX = {latmax}
   NX = {nx}
   NY = {ny}
   NZ = {nz}
   ZMAX_(M) = {zmax}
   !SIGMA_VALUES = 0.0 0.01 0.025 0.05 0.1

   !
"""

fstring_species = """ -------
 SPECIES
 -------
   !
   !     Species of category PARTICLES
   !     NOTE: only one type of PARTICLES species is allowed
   !
   !     TEPHRA    options :  ON / OFF
   !     DUST      options :  ON / OFF
   !
   TEPHRA = {tephra}
   DUST   = {dust}
   !
   !     Species of category AEROSOLS
   !     NOTE: AEROSOLS can run independently or coupled with TEPHRA species.
   !           IF TEPHRA = On it is assumed that the aerosol species are all of magmatic origin its mass fraction
   !           is relative to tephra. If IF TEPHRA = Off then the mass fraction of aerosols must sum 1
   !
   !     H2O                  options : ON / OFF
   !     SO2                  options : ON / OFF
   !     MASS_FRACTION_(%)            : value
   !
   H2O    = {h2o}   MASS_FRACTION_(%) = {h2o_mass_fraction}
   SO2    = {so2}    MASS_FRACTION_(%) = {so2_mass_fraction}
   !
   !     Species of category RADIONUCLIDES
   !     NOTE: all species of RADIONUCLIDES can run simultaneously but are incompatible with a species of category PARTICLES. The
   !     mass fraction of radionuclides must sum 1
   !
   !     CS134                options : ON / OFF
   !     CS137                options : ON / OFF
   !     I131                 options : ON / OFF
   !     SR90                 options : ON / OFF
   !     Y90                  options : ON / OFF
   !     MASS_FRACTION_(%)            : value
   !
   CS134  = {cs134}   MASS_FRACTION_(%) = {cs134_mass_fraction}
   CS137  = {cs137}   MASS_FRACTION_(%) = {cs137_mass_fraction}
   I131   = {i131}   MASS_FRACTION_(%) = {i131_mass_fraction}
   SR90   = {sr90}   MASS_FRACTION_(%) = {sr90_mass_fraction}
   Y90    = {y90}   MASS_FRACTION_(%) = {y90_mass_fraction}
   !
"""

fstring_tephra_tgsd = """ -------------
 TEPHRA_TGSD
 -------------
   !
   !   This block is read by task SetTgsd to generate tephra Total Grain Size Distributions (TGSD).
   !   Note that the effective granulometry can change afterwards if aggrgeation is switched-on or if particle
   !   cut-off (i.e. effective granulometry) are considered
   !
   !   NUMBER_OF_BINS        : value
   !   FI_RANGE              : 2 values
   !   DENSITY_RANGE         : 2 values (linear interpolation)
   !   SPHERICITY_RANGE      : 2 values (linear interpolation)
   !   DISTRIBUTION options  : GAUSSIAN / BIGAUSSIAN / WEIBULL / BIWEIBULL / CUSTOM / ESTIMATE
   !   FI_MEAN               : value. A second value is used if DISTRIBUTION = BIGAUSSIAN
   !   FI_DISP               : value. A second value is used if DISTRIBUTION = BIGAUSSIAN
   !   FI_SCALE              : value. A second value is used if DISTRIBUTION = BIWEIBULL
   !   W_SHAPE               : value. A second value is used if DISTRIBUTION = BIWEIBULL
   !   MIXING_FACTOR         : value. Only used if DISTRIBUTION = BIGAUSSIAN/WIBEIBULL. Default value is 0.5
   !   VISCOSITY_(PAS)       : value. Only used if DISTRIBUTION = ESTIMATE
   !   HEIGHT_ABOVE_VENT_(M) : value. Only used if DISTRIBUTION = ESTIMATE
   !
   NUMBER_OF_BINS   = {number_of_bins}
   FI_RANGE         = {fi_range1} {fi_range2}
   DENSITY_RANGE    = {density_range1} {density_range2}
   SPHERICITY_RANGE = {sphericity_range1}  {sphericity_range2}
   DISTRIBUTION     =  {distribution}
     !
     IF_GAUSSIAN    FI_MEAN  = {gaussian_fi_mean}        FI_DISP = {gaussian_fi_disp}
     IF_BIGAUSSIAN  FI_MEAN  = {bigaussian_fi_mean1} {bigaussian_fi_mean2}  FI_DISP = {bigaussian_fi_disp1} {bigaussian_fi_disp2}  MIXING_FACTOR = {bigaussian_mixing_factor}
     IF_WEIBULL     FI_SCALE = {weibull_fi_scale}        W_SHAPE = {weibull_w_shape}
     IF_BIWEIBULL   FI_SCALE = {biweibull_fi_scale1}  {biweibull_fi_scale2}   W_SHAPE = {biweibull_w_shape1}  {biweibull_w_shape2}  MIXING_FACTOR = {biweibull_mixing_factor}
     IF_CUSTOM      FILE = {custom_file}
     IF_ESTIMATE    VISCOSITY_(PAS) = {estimate_viscosity}  HEIGHT_ABOVE_VENT_(M) = {estimate_height_above_vent}
   !
"""

fstring_radionucleides_tgsd = """ -------------
 CS137_TGSD
 CS134_TGSD
 I131_TGSD
 SR90_TGSD
 Y90_TGSD
 -------------
   !
   !   This block is read by task SetTgsd to generate tephra Total Grain Size Distributions (TGSD).
   !   Note that the effective granulometry can change afterwards if aggrgeation is switched-on or if particle
   !   cut-off (i.e. effective granulometry) are considered
   !
   !   NUMBER_OF_BINS        : value
   !   FI_RANGE              : 2 values
   !   DENSITY_RANGE         : 2 values (linear interpolation)
   !   SPHERICITY_RANGE      : 2 values (linear interpolation)
   !   DISTRIBUTION options  : GAUSSIAN / BIGAUSSIAN / WEIBULL / BIWEIBULL / CUSTOM / ESTIMATE
   !   FI_MEAN               : value. A second value is used if DISTRIBUTION = BIGAUSSIAN
   !   FI_DISP               : value. A second value is used if DISTRIBUTION = BIGAUSSIAN
   !   FI_SCALE              : value. A second value is used if DISTRIBUTION = BIWEIBULL
   !   W_SHAPE               : value. A second value is used if DISTRIBUTION = BIWEIBULL
   !   MIXING_FACTOR         : value. Only used if DISTRIBUTION = BIGAUSSIAN/BIWEIBULL. Default value is 0.5
   !   VISCOSITY_(PAS)       : value. Only used if DISTRIBUTION = ESTIMATE
   !   HEIGHT_ABOVE_VENT_(M) : value. Only used if DISTRIBUTION = ESTIMATE
   !
   NUMBER_OF_BINS   = {number_of_bins}
   FI_RANGE         = {fi_range1} {fi_range2}
   DENSITY_RANGE    = {density_range1} {density_range2}
   SPHERICITY_RANGE = {sphericity_range1}  {sphericity_range2}
   DISTRIBUTION     = {distribution}
     !
     IF_GAUSSIAN    FI_MEAN  = {gaussian_fi_mean}        FI_DISP = {gaussian_fi_disp}
     IF_BIGAUSSIAN  FI_MEAN  = {bigaussian_fi_mean1} {bigaussian_fi_mean2}  FI_DISP = {bigaussian_fi_disp1} {bigaussian_fi_disp2}  MIXING_FACTOR = {bigaussian_mixing_factor}
     IF_WEIBULL     FI_SCALE = {weibull_fi_scale}        W_SHAPE = {weibull_w_shape}
     IF_BIWEIBULL   FI_SCALE = {biweibull_fi_scale1}  {biweibull_fi_scale2}   W_SHAPE = {biweibull_w_shape1}  {biweibull_w_shape2}  MIXING_FACTOR = {biweibull_mixing_factor}
     IF_CUSTOM      FILE = {custom_file}
     IF_ESTIMATE    VISCOSITY_(PAS) = {estimate_viscosity}  HEIGHT_ABOVE_VENT_(M) = {estimate_height_above_vent}
   !
"""



fstring_particle_aggregation  = """ ---------------------
 PARTICLE_AGGREGATION
  --------------------
   !
   !   PARTICLE_CUT_OFF          options : NONE / FI_LARGER_THAN  value / FI_LOWER_THAN  value / D_(MIC)_LARGER_THAN value / D_(MIC)_LOWER_THAN value
   !   AGGREGATION_MODEL         options : NONE / PERCENTAGE / CORNELL / COSTA
   !   NUMBER_OF_AGGREGATE_BINS          : value. Default value is 1. Neglegted if AGGREGATION_MODEL = COSTA
   !   DIAMETER_AGGREGATES_(MIC)         : NUMBER_OF_AGGREGATE_BINS values
   !   DENSITY_AGGREGATES_(KGM3)         : NUMBER_OF_AGGREGATE_BINS values
   !   PERCENTAGE_(%)                    : NUMBER_OF_AGGREGATE_BINS values
   !   VSET_FACTOR                       : value
   !   FRACTAL_EXPONENT                  : value. Only used if AGGREGATION_MODEL = COSTA
   !
   PARTICLE_CUT_OFF          = {particle_cut_off}
   AGGREGATION_MODEL         = {aggregation_model}
   NUMBER_OF_AGGREGATE_BINS  = {number_of_aggregate_bins}
   DIAMETER_AGGREGATES_(MIC) = {diameter_aggregates1} {diameter_aggregates2}
   DENSITY_AGGREGATES_(KGM3) = {density_aggregates1} {density_aggregates2}
   PERCENTAGE_(%)            = {percentage1}   {percentage2}
   VSET_FACTOR               = {vset_factor}
   FRACTAL_EXPONENT          = {fractal_exponent}
   !
"""


fstring_source = """------
SOURCE
------
   !
   !   SOURCE_TYPE                          options : POINT / SUZUKI / TOP-HAT / PLUME / RESUSPENSION
   !   SOURCE_START_(HOURS_AFTER_00) [file_name]    : ndt values (starting times of each source phase)
   !   SOURCE_END_(HOURS_AFTER_00)                  : ndt values (ending   times of each source phase)
   !                                                  NOTE: if optional argument file_name exists, then "start times","end times" and
   !                                                  "source heights" read from the 3-column file_name
   !   LON_VENT                                     : value. Source (vent) longitude
   !   LAT_VENT                                     : value. Source (vent) latitude
   !   VENT_HEIGHT_(M)                              : value. Source (vent) altitude (in m a.s.l.)
   !   HEIGHT_ABOVE_VENT_(M)                        : ndt values
   !   MASS_FLOW_RATE_(KGS)                 options : ndt values / ESTIMATE-MASTIN / ESTIMATE-WOODHOUSE / ESTIMATE-DEGRUYTER
   !
   !   ALFA_PLUME                                   : value (default 0.1). Only used if ESTIMATE-WOODHOUSE / ESTIMATE-DEGRUYTER to estimate MER from H
   !   BETA_PLUME                                   : value (default 0.5). Only used if ESTIMATE-WOODHOUSE / ESTIMATE-DEGRUYTER to estimate MER from H
   !   EXIT_TEMPERATURE_(K)                         : ndt values. Mixture temperature.  Only used if SOURCE_TYPE = PLUME or     to estimate MER from H
   !   EXIT_WATER_FRACTION_(%)                      : ndt values. Total water fraction. Only used if SOURCE_TYPE = PLUME or     to estimate MER from H
   !
   !                                                 The following variables in IF_SUZUKI_SOURCE block are read only if SOURCE_TYPE = SUZUKI
   !                                                 ---------------------------------------------------------------------------------------
   !   A                                            : ndt values. A-Suzuki parameter
   !   L                                            : ndt values. L-Suzuki parameter
   !
   !                                                 The following variables in IF_TOP-HAT_SOURCE block are read only if SOURCE_TYPE = HAT
   !                                                 -------------------------------------------------------------------------------------
   !   THICKNESS_(M)                                : ndt values. Hat thickness
   !
   !                                                 The following variables in IF_PLUME_SOURCE block are read only if SOURCE_TYPE = PLUME
   !                                                 --------------------------------------------------------------------------------------
   !   SOLVE_PLUME_FOR                      options : MFR / HEIGHT
   !   MFR_SEARCH_RANGE                             : 2 values n1 and n2, where: 10**n1 < MFR < 10**n2. Only used SOLVE_PLUME_FOR = MFR
   !   EXIT_VELOCIY_(MS)                            : ndt values
   !   EXIT_GAS_WATER_TEMPERATURE_(K)               : ndt values. Optional. If not given, assumed equal to EXIT_TEMPERATURE_(K)
   !   EXIT_LIQUID_WATER_TEMPERATURE_(K)            : ndt values. Optional. If not given, assumed equal to EXIT_TEMPERATURE_(K)
   !   EXIT_SOLID_WATER_TEMPERATURE_(K)             : ndt values. Optional. If not given, assumed equal to EXIT_TEMPERATURE_(K)
   !   EXIT_GAS_WATER_FRACTION_(%)                  : ndt values. Optional. If not given, assumed equal to EXIT_WATER_FRACTION_(%)
   !   EXIT_LIQUID_WATER_FRACTION_(%)               : ndt values. Optional. If not given, assumed equal to 0
   !   EXIT_SOLID_WATER_FRACTION_(%)                : ndt values. Optional. If not given, assumed equal to 0
   !   WIND_COUPLING                        options : yes/no. If NO, Ua=0 is assumed
   !   AIR_MOISTURE                         options : yes/no. If NO, wa=0 is assumed (dry entrained air only)
   !   REENTRAINMENT                        options : yes/no. If NO, particle reentrainment is neglected (f=0)
   !   LATENT_HEAT                          options : yes/no. If NO, latent heat contribution is neglected
   !   BURSIK_FACTOR                                : value. Bursik factor xi. If not given, assumed equal to 0.1
   !   Z_MIN_WIND                                   : value. Ignore wind entrainment below this zvalue (low jet region). If not given, assumed equal to 100
   !   C_UMBRELLA                                   : value. Thickness of umbrella region relative to Hb (>1). If not given, assumed equal to 1.32
   !   A_S                                  options : CONSTANT (value jet, value plume) / KAMINSKI-R / KAMINSKI-C / OLD
   !   A_V                                  options : CONSTANT (value) / TATE
   !
   SOURCE_TYPE                   = {source_type}
   SOURCE_START_(HOURS_AFTER_00) = {source_start}
   SOURCE_END_(HOURS_AFTER_00)   = {source_end}
   !
   LON_VENT        = {lon_vent}
   LAT_VENT        = {lat_vent}
   VENT_HEIGHT_(M) = {vent_height}
   !
   HEIGHT_ABOVE_VENT_(M)         = {height_above_vent}
   MASS_FLOW_RATE_(KGS)          = {mass_flow_rate}
   ALFA_PLUME                    = {alfa_plume}
   BETA_PLUME                    = {beta_plume}
   EXIT_TEMPERATURE_(K)          = {exit_temperature}
   EXIT_WATER_FRACTION_(%)       = {exit_water_fraction}
   !
   IF_SUZUKI_SOURCE
      A = {a1} {a2}
      L = {l}
   !
   IF_TOP-HAT_SOURCE
      THICKNESS_(M) = {thickness} 
   !
   IF_PLUME_SOURCE
      SOLVE_PLUME_FOR                   = {solve_plume_for}
      MFR_SEARCH_RANGE                  = {mfr_search_range1}  {mfr_search_range2}
      EXIT_VELOCITY_(MS)                = {exit_velocity}
      EXIT_GAS_WATER_TEMPERATURE_(K)    = {exit_gas_water_temperature}
      EXIT_LIQUID_WATER_TEMPERATURE_(K) = {exit_liquid_water_temperature}
      EXIT_SOLID_WATER_TEMPERATURE_(K)  = {exit_solid_water_temperature}
      EXIT_GAS_WATER_FRACTION_(%)       = {exit_gas_water_fraction}
      EXIT_LIQUID_WATER_FRACTION_(%)    = {exit_liquid_water_fraction}
      EXIT_SOLID_WATER_FRACTION_(%)     = {exit_solid_water_fraction}
      WIND_COUPLING = {wind_coupling}
      AIR_MOISTURE  = {air_moisture}
      LATENT_HEAT   = {latent_heat}
      REENTRAINMENT = {reentrainment}
      BURSIK_FACTOR = {bursik_factor}
      Z_MIN_WIND    = {z_min_wind}
      C_UMBRELLA    = {c_umbrella}
      A_S           = {a_s}
      A_V           = {a_v}
   !
"""

fstring_ensemble = """ --------
 ENSEMBLE
 --------
   !
   !  RANDOM_NUMBERS_FROM_FILE                   options  : YES/NO (default NO)
   !  PERTURBATE_COLUMN_HEIGHT                   options  : NO / RELATIVE / ABSOLUTE
   !  PERTURBATE_MASS_FLOW_RATE                  options  : NO / RELATIVE / ABSOLUTE (only if MASS_FLOW_RATE_(KGS) is given; desactivated for PLUME or ESTIMATE options)
   !  PERTURBATE_SOURCE_START                    options  : NO / RELATIVE / ABSOLUTE  
   !  PERTURBATE_SOURCE_DURATION                 options  : NO / RELATIVE / ABSOLUTE  
   !  PERTURBATE_TOP-HAT_THICKNESS               options  : NO / RELATIVE / ABSOLUTE (only if SOURCE_TYPE = TOP-HAT) 
   !  PERTURBATE_SUZUKI_A                        options  : NO / RELATIVE / ABSOLUTE (only if SOURCE_TYPE = SUZUKI) 
   !  PERTURBATE_SUZUKI_L                        options  : NO / RELATIVE / ABSOLUTE (only if SOURCE_TYPE = SUZUKI) 
   !  PERTURBATE_WIND                            options  : NO / RELATIVE / ABSOLUTE
   !  PERTURBATE_DATA_INSERTION_CLOUD_HEIGHT     options  : NO / RELATIVE / ABSOLUTE (only if INITIAL_CONDITION = INSERTION)
   !  PERTURBATE_DATA_INSERTION_CLOUD_THICKNESS  options  : NO / RELATIVE / ABSOLUTE (only if INITIAL_CONDITION = INSERTION)
   !  PERTURBATE_FI_MEAN                         options  : NO / RELATIVE / ABSOLUTE (only if DISTRIBUTION = GAUSSIAN / BIGAUSSIAN)
   !  PERTURBATE_DIAMETER_AGGREGATES             options  : NO / RELATIVE / ABSOLUTE 
   !  PERTURBATE_DENSITY_AGGREGATES              options  : NO / RELATIVE / ABSOLUTE 
   !  (for all PERTURBATE options above)
   !     PERTURBATION_RANGE                      options  : value. Percentage (in %) for RELATIVE or value (in units) for ABSOLUTE 
   !     PDF                                     options  : UNIFORM / GAUSSIAN   
   !  POSTPROCESS_MEMBERS                        options  : YES/NO (default NO)
   !  POSTPROCESS_MEAN                           options  : YES/NO (default NO)
   !  POSTPROCESS_MEDIAN                         options  : YES/NO (default NO)
   !  POSTPROCESS_STANDARD_DEV                   options  : YES/NO (default NO)
   !  POSTPROCESS_PROBABILITY                    options  : YES/NO (default NO)
   !  POSTPROCESS_PERCENTILES                    options  : YES/NO (default NO)
   !  IF_POSTPROCESS_PROBABILITY 
   !    CONCENTRATION_THRESHOLDS_(MG/M3)                  : value(s)
   !    COLUMN_MASS_THRESHOLDS_(G/M2)                     : value(s)
   !    COLUMN_MASS_THRESHOLDS_(DU)                       : value(s)
   !    GROUND_LOAD_THRESHOLDS_(KG/M2)                    : value(s)
   !  IF_POSTPROCESS_PERCENTILES
   !    PERCENTILE_VALUES_(%)                             : value(s)
   !
   RANDOM_NUMBERS_FROM_FILE  = {random_numbers_from_file}
   !
   PERTURBATE_COLUMN_HEIGHT  = {perturbate_column_height}
   IF_PERTURBATE_COLUMN_HEIGHT
      PERTURBATION_RANGE = {column_height_perturbation_range}
      PDF                = {column_height_pdf}
   !
   PERTURBATE_MASS_FLOW_RATE = {perturbate_mass_flow_rate}
   IF_PERTURBATE_MASS_FLOW_RATE
      PERTURBATION_RANGE = {mass_flow_rate_perturbation_range}
      PDF                = {mass_flow_rate_pdf}
   !
   PERTURBATE_SOURCE_START = {perturbate_source_start}
   IF_PERTURBATE_SOURCE_START
      PERTURBATION_RANGE = {perturbate_source_start_range}
      PDF                = {perturbate_source_start_pdf}
   !
   PERTURBATE_SOURCE_DURATION = {perturbate_source_duration}
   IF_PERTURBATE_SOURCE_DURATION
      PERTURBATION_RANGE = {perturbate_source_duration_range}
      PDF                = {perturbate_source_duration_pdf}
   !
   PERTURBATE_TOP-HAT_THICKNESS = {perturbate_top_hat_thickness}
   IF_PERTURBATE_TOP-HAT_THICKNESS
      PERTURBATION_RANGE = {perturbate_top_hat_thickness_range}
      PDF                = {perturbate_top_hat_pdf}
   !
   PERTURBATE_SUZUKI_A = {perturbate_suzuki_a}
   IF_PERTURBATE_SUZUKI_A
      PERTURBATION_RANGE = {perturbate_suzuki_a_range}
      PDF                = {perturbate_suzuki_a_pdf}
   !
   PERTURBATE_SUZUKI_L = {perturbate_suzuki_l}
   IF_PERTURBATE_SUZUKI_L
      PERTURBATION_RANGE = {perturbate_suzuki_l_range}
      PDF                = {perturbate_suzuki_l_pdf}
   !
   PERTURBATE_WIND = {perturbate_wind}
   IF_PERTURBATE_WIND
      PERTURBATION_RANGE = {perturbate_wind_range}
      PDF                = {perturbate_wind_pdf}
   !
   PERTURBATE_DATA_INSERTION_CLOUD_HEIGHT  = {perturbate_data_insertion_cloud_height}
   IF_PERTURBATE_DATA_INSERTION_CLOUD_HEIGHT
      PERTURBATION_RANGE = {perturbate_data_insertion_cloud_height_range}
      PDF                = {perturbate_data_insertion_cloud_height_pdf}
   !
   PERTURBATE_DATA_INSERTION_CLOUD_THICKNESS = {perturbate_data_insertion_cloud_thickness}
   IF_PERTURBATE_DATA_INSERTION_CLOUD_THICKNESS
      PERTURBATION_RANGE = {perturbate_data_insertion_cloud_thickness_range}
      PDF                = {perturbate_data_insertion_cloud_thickness_pdf}
   !
   PERTURBATE_FI_MEAN = {perturbate_fi_mean}
   IF_PERTURBATE_FI_MEAN
      PERTURBATION_RANGE = {perturbate_fi_range}
      PDF                = {perturbate_fi_pdf}
   !
   PERTURBATE_DIAMETER_AGGREGATES_(MIC) = {perturbate_diamater_aggregates}
   IF_PERTURBATE_DIAMETER_AGGREGATES_(MIC)
      PERTURBATION_RANGE = {perturbate_diamater_aggregates_range}
      PDF                = {perturbate_diamater_aggregates_pdf}
   !
   PERTURBATE_DENSITY_AGGREGATES = {perturbate_density_aggregates}
   IF_PERTURBATE_DENSITY_AGGREGATES
      PERTURBATION_RANGE = {perturbate_density_aggregates_range}
      PDF                = {perturbate_density_aggregates_pdf}
   !
"""

fstring_ensemble_postprocess = """ --------------------
 ENSEMBLE_POSTPROCESS
 --------------------
   !
   !  POSTPROCESS_MEMBERS                   options  : YES/NO. If save individual members
   !  POSTPROCESS_MEAN                      options  : YES/NO. If compute ensemble mean
   !  POSTPROCESS_LOGMEAN                   options  : YES/NO. If compute ensemble mean after a log transformation
   !  POSTPROCESS_MEDIAN                    options  : YES/NO. If compute ensemble median
   !  POSTPROCESS_STANDARD_DEV              options  : YES/NO. If compute ensemble standard deviation
   !  POSTPROCESS_PROBABILITY               options  : YES/NO. If compute exceedance probabilities
   !  POSTPROCESS_PERCENTILES               options  : YES/NO. If compute percentiles
   !
   POSTPROCESS_MEMBERS      = {postprocess_members}
   POSTPROCESS_MEAN         = {postprocess_mean}
   POSTPROCESS_LOGMEAN      = {postprocess_logmean}
   POSTPROCESS_MEDIAN       = {postprocess_median}      
   POSTPROCESS_STANDARD_DEV = {postprocess_standard_dev}
   POSTPROCESS_PROBABILITY  = {postprocess_probability}
   POSTPROCESS_PERCENTILES  = {postprocess_percentiles}
   !
   IF_POSTPROCESS_PROBABILITY 
      CONCENTRATION_THRESHOLDS_(MG/M3) = {postprocess_probability_concentration_thresholds}
      COLUMN_MASS_THRESHOLDS_(G/M2)    = {postprocess_probability_column_mass_thresholds_gm2}
      COLUMN_MASS_THRESHOLDS_(DU)      = {postprocess_probability_column_mass_thresholds_du}
      GROUND_LOAD_THRESHOLDS_(KG/M2)   = {postprocess_probability_ground_load_thresholds} 
   !
   IF_POSTPROCESS_PERCENTILES
      PERCENTILE_VALUES_(%) = {postprocess_percentiles_percentile_values} 
   !
"""

fstring_model_physics = """ -------------
 MODEL_PHYSICS
 -------------
   !
   !  LIMITER                               options  : MINMOD  / SUPERBEE / OSPRE
   !  TIME_MARCHING                         options  : EULER / RUNGE-KUTTA
   !  CFL_CRITERION                         options  : ONE_DIMENSIONAL / ALL_DIMENSIONS
   !  CFL_SAFETY_FACTOR                              : value (default 0.9)
   !
   !  TERMINAL_VELOCITY_MODEL               options  : ARASTOOPOUR / GANSER / WILSON / DELLINO / PFEIFFER / DIOGUARDI2017 / DIOGUARDI2018
   !  HORIZONTAL_TURBULENCE_MODEL           options  : CONSTANT = value / CMAQ / RAMS
   !  VERTICAL_TURBULENCE_MODEL             options  : CONSTANT = value / SIMILARITY
   !  RAMS_CS                                        : value. Only used if HORIZONTAL_TURBULENCE_MODEL = RAMS 
   !  WET_DEPOSITION                        options  : YES/NO
   !  DRY_DEPOSITION                        options  : YES/NO
   !  GRAVITY_CURRENT                       options  : YES/NO. Activate GC  based on Suzuki & Koyaguchi (2009)
   !  C_FLOW_RATE                                    : value. Only read if GRAVITY_CURRENT = YES (0.43d3 for tropical eruptions; 0.87d3 for mid-latitude and polar).
   !  LAMBDA_GRAV                                    : value. Only read if GRAVITY_CURRENT = YES
   !  K_ENTRAIN                                      : value. Only read if GRAVITY_CURRENT = YES
   !  BRUNT_VAISALA                                  : value. Only read if GRAVITY_CURRENT = YES
   !  GC_START_(HOURS_AFTER_00)                      : value. Only read if GRAVITY_CURRENT = YES. GC start time
   !  GC_END_(HOURS_AFTER_00)                        : value. Only read if GRAVITY_CURRENT = YES. GC end   time
   !
   LIMITER                     = {limiter}
   TIME_MARCHING               = {time_marching}
   CFL_CRITERION               = {cfl_criterion}
   CFL_SAFETY_FACTOR           = {cfl_safety_factor}
   !
   TERMINAL_VELOCITY_MODEL     = {terminal_velocity_model}
   HORIZONTAL_TURBULENCE_MODEL = {horizontal_turbulence_model}
   VERTICAL_TURBULENCE_MODEL   constant = {vertical_turbulence_model} ! SIMILARITY
   RAMS_CS                     = {rams_cs}
   WET_DEPOSITION              = {wet_deposition}
   DRY_DEPOSITION              = {dry_deposition}
   GRAVITY_CURRENT             = {gravity_current}
   !
   IF_GRAVITY_CURRENT
      C_FLOW_RATE               = {c_flow_rate}
      LAMBDA_GRAV               = {lambda_grav}
      K_ENTRAIN                 = {k_entrain}
      BRUNT_VAISALA             = {brunt_vaisala}
      GC_START_(HOURS_AFTER_00) = {gc_start}
      GC_END_(HOURS_AFTER_00)   = {gc_end}
   !
"""

fstring_model_output = """ -------------
 MODEL_OUTPUT
 -------------
   !
   !  PARALLEL_IO                        options  : YES/NO
   !  LOG_FILE_LEVEL                     options  : NONE / NORMAL / FULL
   !  RESTART_TIME_INTERVAL_(HOURS)      options  : NONE / END_ONLY / value
   !  OUTPUT_INTERMEDIATE_FILES          options  : YES/NO. Outputs intermediate files (can be set to NO only for task ALL)
   !  OUTPUT_TIME_START_(HOURS)                   : RUN_START / value
   !  OUTPUT_TIME_INTERVAL_(HOURS)                : value
   !  OUTPUT_3D_CONCENTRATION            options  : YES/NO. Outputs total concentration on sigma planes (summ over all bins of a given substance)
   !  OUTPUT_3D_CONCENTRATION_BINS       options  : YES/NO. Outputs bin   concentration on sigma planes (          all bins of a given substance)
   !  OUTPUT_SURFACE_CONCENTRATION       options  : YES/NO. Outputs concentration at surface            (summ over all bins of a given substance)
   !  OUTPUT_COLUMN_LOAD                 options  : YES/NO. Outputs column  mass load                   (summ over all bins of a given substance)
   !  OUTPUT_CLOUD_TOP                   options  : YES/NO. Outputs cloud top heigh
   !  OUTPUT_GROUND_LOAD                 options  : YES/NO. Outputs deposit mass load                   (summ over all bins of a given substance)
   !  OUTPUT_GROUND_LOAD_BINS            options  : YES/NO. Outputs deposit mass load                   (          all bins of a given substance)
   !  OUTPUT_WET_DEPOSITION              options  : YES/NO. Outputs wet deposition mass                 (summ over all bins of a given substance)
   !  TRACK_POINTS                       options  : YES/NO.
   !  TRACK_POINTS_FILE                           : File with the list of tracked points
   !  OUTPUT_CONCENTRATION_AT_XCUTS      options  : YES/NO. If yes, outputs cuts at specifyed X-values
   !  OUTPUT_CONCENTRATION_AT_YCUTS      options  : YES/NO. If yes, outputs cuts at specifyed Y-values
   !  OUTPUT_CONCENTRATION_AT_ZCUTS      options  : YES/NO. If yes, outputs cuts at specifyed Z-values
   !  OUTPUT_CONCENTRATION_AT_FL         options  : YES/NO. If yes, outputs cuts at specifyed FLs
   !
   PARALLEL_IO                   = {parallel_io}
   LOG_FILE_LEVEL                = {log_file_level}
   RESTART_TIME_INTERVAL_(HOURS) = {restart_time_interval}
   OUTPUT_INTERMEDIATE_FILES     = {output_intermediate_files}
   OUTPUT_TIME_START_(HOURS)     = {output_time_start}
   OUTPUT_TIME_INTERVAL_(HOURS)  = {output_time_interval}
   OUTPUT_3D_CONCENTRATION       = {output_3d_concentration}
   OUTPUT_3D_CONCENTRATION_BINS  = {output_3d_concentration_bins}
   OUTPUT_SURFACE_CONCENTRATION  = {output_surface_concentration}
   OUTPUT_COLUMN_LOAD            = {output_column_load}
   OUTPUT_CLOUD_TOP              = {output_cloud_top}
   OUTPUT_GROUND_LOAD            = {output_ground_load}
   OUTPUT_GROUND_LOAD_BINS       = {output_ground_load_bins}
   OUTPUT_WET_DEPOSITION         = {output_wet_deposition}
   TRACK_POINTS                  = {output_track_points}
   TRACK_POINTS_FILE             = {output_track_points_file}
   !
   OUTPUT_CONCENTRATION_AT_XCUTS = {output_concentrations_at_xcuts}
      X-VALUES  {x_values}
   OUTPUT_CONCENTRATION_AT_YCUTS = {output_concentrations_at_ycuts}
      Y-VALUES  {y_values}
   OUTPUT_CONCENTRATION_AT_ZCUTS = {output_concentrations_at_zcuts}
      Z-VALUES  {z_values} 
   OUTPUT_CONCENTRATION_AT_FL    = {output_concentrations_at_fl}
      FL-VALUES  {fl_values}
   !
"""

fstring_model_validation = """ ----------------
 MODEL_VALIDATION
 ----------------
   !
   !  OBSERVATIONS_TYPE                         options  : SATELLITE_DETECTION / SATELLITE_RETRIEVAL / DEPOSIT_CONTOURS / DEPOSIT_POINTS
   !  OBSERVATIONS_FILE                         options  : path to the observations file 
   !  OBSERVATIONS_DICTIONARY_FILE              options  : optional. Path to the dictionary file. If not given, dafault value is used.
   !  RESULTS_FILE                              options  : path to the FALL3D results file (both name.res.nc or name.ens.nc are supported)
   !  COLUMN_MASS_OBSERVATION_THRESHOLD_(G/M2)  options  : value. Default 0.1
   !  COLUMN_MASS_OBSERVATION_THRESHOLD_(DU)    options  : value. Default 1
   !  GROUND_LOAD_OBSERVATION_THRESHOLD_(KG/M2) options  : value. Default 0.1
   !
   OBSERVATIONS_TYPE            = {observations_type}
   OBSERVATIONS_FILE            = {observations_file}
   OBSERVATIONS_DICTIONARY_FILE = {observations_dictionary_file}
   RESULTS_FILE                 = {results_file}
   !
   IF_OBSERVATIONS_TYPE_SATELLITE
      COLUMN_MASS_OBSERVATION_THRESHOLD_(G/M2)  = {column_mass_observations_threshold_gm2}
      COLUMN_MASS_OBSERVATION_THRESHOLD_(DU)    = {column_mass_observations_threshold_du}
   !
   IF_OBSERVATIONS_TYPE_DEPOSIT
      GROUND_LOAD_OBSERVATION_THRESHOLD_(KG/M2) = {ground_load_observation_threshold_kgm2} 
"""

