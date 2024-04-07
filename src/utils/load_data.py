import pandas as pd
from .data_processing import fourier_series, add_exogenous_variables, clean_data, loadData, add_time_features
import scipy.signal as signal
import numpy as np
from .clean_ukdata import combine_training_datasets, pv_anomalies_to_nan, split_X_y_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils.data_processing import compute_netload_ghi
import scipy.signal as signal

def load_hybrid_res_data(data_path='../../Dataset/IEEECompetition/pv_wind_gen_data.csv', sun_rise=5, sun_set=17):

    data =pd.read_csv(data_path)
    data.rename(columns={'dtm': 'timestamp',   'Wind_MWh_credit': 'WindGen(MWh)', 'Solar_MWh_credit': 'PVGen(MWh)'}, inplace=True)
    data=data[['timestamp', 'WindSpeed', 'Radiation',  'Season','WindGen(MWh)', 'PVGen(MWh)']]
    data.index.name = 'timestamp' 
    data = data.set_index("timestamp")
    
    data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
    data = data.resample(rule='30T', closed='left', label='right').mean()
    data = data[~data.index.duplicated(keep='first')]
    data = data.asfreq('30min')
    data = data.sort_index()
    
    # Forward fill missing values
    data.ffill(inplace=True)
    
    data.index.name = 'timestamp'
    data = add_exogenous_variables(data.reset_index(), one_hot=False)
    data = data.set_index("timestamp").copy()
    
    def day_night(x):
        if (x > sun_rise) and (x <= sun_set):
            return 1
        else:
            return 0
    
    data['Session']=data['HOUR'].apply(day_night).values.astype(int)
    return data


# Load weather data
def load_solar_radiation_PT():
    data = pd.DataFrame()
    for path in ['../../Dataset/PT/solar_cast_2019.csv',
                 '../../Dataset/PT/solar_cast_2020.csv',
                 '../../Dataset/PT/solar_cast_2020_2022.csv',
                 '../../Dataset/PT/solar_cast_historical_2022.csv',
                 '../../Dataset/PT/solar_cas_forecast_2022.csv']:
        data_temp = pd.read_csv(path)
        data_temp['Date'] = pd.to_datetime(
            data_temp['PeriodEnd'] if 'PeriodEnd' in data_temp.columns else data_temp['period_end'])
        data_temp = data_temp.drop(
            ['PeriodStart', 'Period', 'PeriodEnd'] if 'PeriodEnd' in data_temp.columns else ['period_end', 'period'],
            axis=1)

        if 'PeriodEnd' not in data_temp.columns:
            data_temp.rename(
                columns={'ghi': 'Ghi', 'ebh': 'Ebh', 'dni': 'Dni', 'dhi': 'Dhi', 'cloud_opacity': 'CloudOpacity'},
                inplace=True)
        # data_temp=data_temp[['Ghi', 'Ebh', 'Dni', 'Dhi', 'CloudOpacity', 'Date']]
        data = pd.concat([data, data_temp], ignore_index=True)[
            ['Ghi', 'Ebh', 'Dni', 'Dhi', 'CloudOpacity', 'AirTemp', 'Date']]

    data.set_index('Date', inplace=True)
    data.index = data.index.tz_convert("Atlantic/Madeira")
    # Aggregating in 1H intervals
    # ==============================================================================
    # The Date column is eliminated so that it does not generate an error when aggregating.
    # The Holiday column does not generate an error since it is Boolean and is treated as 0-1.
    data = data.resample(rule='30T', closed='left', label='right').mean()
    data = data[~data.index.duplicated(keep='first')]
    data = data.asfreq('30min')
    data = data.sort_index()
    return data


def load_net_load_power_PT():
    data = pd.DataFrame()

    for path in ['../../Dataset/PT/substation_all_phases.csv',
                 '../../Dataset/PT/substation_2021_2022.csv',
                 '../../Dataset/PT/substation_2022.csv']:
        data_temp = pd.read_csv(path)
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp'])

        if 'measure_grid' not in data_temp.columns:
            data_temp.rename(columns={'measure_cons': 'measure_grid'}, inplace=True)

        data_temp = data_temp[['timestamp', 'measure_grid']]
        data_temp.rename(columns={'measure_grid': 'Load'}, inplace=True)

        data = pd.concat([data, data_temp], ignore_index=True)

    data.set_index('timestamp', inplace=True)
    data.index = data.index.tz_convert("Atlantic/Madeira")
    # Aggregating in 1H intervals
    # ==============================================================================
    # The Date column is eliminated so that it does not generate an error when aggregating.
    # The Holiday column does not generate an error since it is Boolean and is treated as 0-1.
    data = data.resample(rule='30T', closed='left', label='right').mean()
    data = data[~data.index.duplicated(keep='first')]
    data = data.asfreq('30min')
    data = data.sort_index()

    return data


def load_substation_data(add_ghi=False, SAMPLES_PER_DAY=48, interpolate_missing_values=True):
    """
    Load an already cleaned version of the dataset
    """
    load = load_net_load_power_PT()
    load = load.ffill(limit=2)

    radiation = load_solar_radiation_PT()
    radiation = radiation.ffill(limit=2)

    print(f"Total data sample: {load.shape[0]}")
    print(f"Missing data sample: {load.isnull().sum()[0]}")
    print(f" percentage of Missing data sample: {load.isnull().sum()[0] / len(load)}")

    # convernt power into kW
    load['Load'] = load['Load'].values / 1000
    data = pd.merge(radiation, load, how='outer', left_index=True, right_index=True)
    data = clean_data(data, SAMPLES_PER_DAY=SAMPLES_PER_DAY, columns=['Load'])
    data.sort_index(inplace=True)

    print(" ")
    print(f"Total data sample after cleaning: {data.shape[0]}")
    print(f"Missing data sample after cleaning: {data.isnull().sum()[0]}")
    print(f" percentage of data loss : {1 - (data.shape[0] / len(load))}")

    # data.index = data.index.astype('datetime64[ns]')
    data.index.name = 'timestamp'
    min_dt = min([data.index.min()])
    max_dt = max([data.index.max()]) + pd.Timedelta(minutes=30)
    dt_rng = pd.date_range(min_dt, max_dt, freq='30T')
    data = data.reindex(dt_rng)

    data.index.name = 'timestamp'
    data = add_exogenous_variables(data.reset_index(), one_hot=False)
    data = data.set_index("timestamp")

    # remove anamoly airtime data
    for col in ['AirTemp', 'CloudOpacity', 'Ebh', 'Dni', 'Dhi', 'Ghi']:
        data[col]['2022-'] = np.where(data[col]['2022-'].diff() == 0.0, np.nan, data[col]['2022-'])

    ##interpolate missing values with RFregressor
    if interpolate_missing_values:
        columns = ['AirTemp', 'CloudOpacity', 'Ebh', 'Dni', 'Dhi', 'Ghi', 'Load', ]
        features = [
            'DAYOFWEEK', 'WEEK', 'DAYOFYEAR', 'MONTH', 'DAY', 'HOUR', 'WEEKDAY',
            'WEEKEND', 'SATURDAY', 'SUNDAY']
        for column in columns:
            df_feats = data[features + [column]]

            missing_site_power_dts = data.index[data[column].isnull()]
            missing_dt_X = df_feats.loc[missing_site_power_dts].drop(column, axis=1).values
            X, y = split_X_y_data(df_feats, column)
            model = RandomForestRegressor()
            model.fit(X, y)

            data.loc[missing_site_power_dts, column] = model.predict(missing_dt_X)
            features.append(column)
    else:
        data = data.dropna()
    ##fill one week missing data in march 2022 with data of previous year
    # for column in ['Ghi', 'Ebh', 'Dni', 'Dhi', 'CloudOpacity']:
    # data[column]['2022-02-27': '2022-03-11']=data[column]['2021-02-27': '2021-03-11']

    data.rename(columns={'AirTemp': 'Temperature', 'Load': 'NetLoad'}, inplace=True)
    data = add_time_features(data, hemisphere='Northern')
    data['Season'] = pd.Categorical(data['Season'])
    data['Season'] = data['Season'].astype('category').cat.codes
    data['NetLoad'] = signal.medfilt(data['NetLoad'].values, 3)

    if add_ghi:
        data['NetLoad-Ghi'] = np.nan
        net_load_ghi = compute_netload_ghi(data['NetLoad'].values, data['Ghi'].values, SAMPLES_PER_DAY).flatten()
        data = data.iloc[len(data) - len(net_load_ghi):]
        data['NetLoad-Ghi'] = net_load_ghi
        del net_load_ghi

    # data=add_time_features(data, hemisphere = 'Northern')

    return data


def load_uk_data(data_path='../../Dataset/UK', add_ghi=False, SAMPLES_PER_DAY=48):
    data = combine_training_datasets('../../Dataset/UK/train/').interpolate(limit=1)
    data = clean_data(data, SAMPLES_PER_DAY=48, columns=['demand_MW'])
    data.drop(['demand', 'pv', 'weather'], axis=1, inplace=True)
    data.index.name = 'timestamp'
    min_dt = min([data.index.min()])
    max_dt = max([data.index.max()]) + pd.Timedelta(minutes=30)
    dt_rng = pd.date_range(min_dt, max_dt, freq='30T')
    data = data.reindex(dt_rng)

    data.index.name = 'timestamp'
    data = add_exogenous_variables(data.reset_index(), one_hot=False)
    data = data.set_index("timestamp").copy()
    # d that two weeks in May 2018 and a couple of days
    data['demand_MW']['2018-05-07':'2018-05-13'] = data['demand_MW']['2018-04-30':'2018-05-06'].values
    data['demand_MW']['2018-11-03':'2018-11-05'] = data['demand_MW']['2018-11-26':'2018-11-28'].values
    data['demand_MW']['2020-3-17 06':'2020-3-17 16'] = data['demand_MW']['2020-3-9 06':'2020-3-9 16'].values

    # Adding temperature data
    temp_loc_cols = data.columns[data.columns.str.contains('temp_location')]
    data.loc[data.index, temp_loc_cols] = data[temp_loc_cols].copy()
    data = data.ffill(limit=1)
    data['Temperature'] = data[[f'temp_location{i}' for i in range(1, 7)]].mean(1)
    data['Temperature-min'] = data[[f'temp_location{i}' for i in range(1, 7)]].min(1)
    data['Temperature-max'] = data[[f'temp_location{i}' for i in range(1, 7)]].max(1)

    # Adding solar irradiance data
    solar_loc_cols = data.columns[data.columns.str.contains('solar_location')]
    data.loc[data.index, solar_loc_cols] = data[solar_loc_cols].copy()
    data = data.ffill(limit=1)
    data['Ghi'] = data[[f'solar_location{i}' for i in range(1, 7)]].mean(1)
    data['Ghi-min'] = data[[f'solar_location{i}' for i in range(1, 7)]].min(1)
    data['Ghi-max'] = data[[f'solar_location{i}' for i in range(1, 7)]].min(1)
    data.rename(columns={'pv_power_mw': 'PV', 'demand_MW': 'Load'}, inplace=True)
    data['NetLoad'] = data['Load'] - data['PV']
    # df_features['pv_7d_lag'] = df['pv_power_mw'].rolling(48*7).mean().shift(48*7)
    data = data.ffill(limit=1)
    data = add_time_features(data, hemisphere='Northern')
    data['Season'] = pd.Categorical(data['Season'])
    data['Season'] = data['Season'].astype('category').cat.codes
    data['NetLoad'] = signal.medfilt(data['NetLoad'].values, 3)

    if add_ghi:
        data['NetLoad-Ghi'] = np.nan
        net_load_ghi = compute_netload_ghi(data['NetLoad'].values, data['Ghi'].values, SAMPLES_PER_DAY).flatten()
        data = data.iloc[len(data) - len(net_load_ghi):]
        data['NetLoad-Ghi'] = net_load_ghi
        del net_load_ghi
    return data


def load_albania_data(data_path='../../Dataset/Albania/Year_2016_2017_2018__2019_data_.xlsx', samples_per_day=48):
    data = pd.read_excel(data_path, sheet_name=0)
    data = clean_data(data, SAMPLES_PER_DAY=samples_per_day, columns=['Load'])
    data.rename(columns={'Time': 'timestamp', 'Time_double': 'TimeDouble', 'Load': 'NetLoad', 'full_temp': 'FullTemp',
                         'full_humid': 'FullHumid', 'full_rain': 'FullRain', 'Muaj': 'Month', 'Ditet': 'DayOfWeek',
                         'Ore': 'Hour', 'Pushimet': 'Holiday', 'Oret e nates': 'NightHour',
                         'Pushimet vjetore': 'AnnualHoliday', 'Oret e nates_2': 'NightHour_2',
                         'Oret e pikut': 'PeakHour'}, inplace=True)
    data['Day'] = data['timestamp'].dt.day
    data.index.name = 'index'
    return data


def load_austgrid(customers=[13, 14, 20, 33, 35, 38, 39, 56, 69, 73, 74, 75,
                             82, 87, 88, 101, 104, 106, 109, 110, 119, 124, 130, 137, 141,
                             144, 152, 153, 157, 161, 169, 176, 184, 188, 189, 193, 201, 202,
                             204, 206, 207, 210, 211, 212, 214, 218, 244, 246, 253, 256, 273,
                             276, 297]):
    df = pd.read_csv('../../Dataset/Austgrid/Processed_energy_Dataset.csv')
    df['NetLoad'] = df['Energy_Consumption_kWh'] - df['Solar_Production_kWh']
    df = df.set_index('timestamp').sort_index()

    total = []
    for c in customers:
        x = df[df.Customer == c][['NetLoad']]
        total.append(x)

    df_netload = pd.concat(total, 1)
    df_netload.columns = [str(c) for c in customers]

    df_netload = df_netload[~df_netload.index.duplicated(keep='first')]
    df_netload['NetLoad'] = df_netload[[str(c) for c in customers]].sum(axis=1)
    df_netload.index = pd.to_datetime(df_netload.index)
    df_netload = df_netload.resample(rule='30T', closed='left', label='right').mean()
    df_netload = df_netload.asfreq('30min')
    df_netload = df_netload.sort_index()
    df_netload = add_time_features(df_netload, hemisphere='Southern')
    df_netload = add_exogenous_variables(df_netload.reset_index(), one_hot=False)
    df_netload = df_netload.set_index("timestamp")
    return df_netload


def load_irradiance_data():
    data = pd.DataFrame()
    for path in ['../../Dataset/PT/solar_cast_2019.csv', '../../Dataset/PT/solar_cast_2020.csv',
                 '../../Dataset/PT/solar_cast_2020_2022.csv']:
        data_temp = pd.read_csv(path)
        data_temp['Date'] = pd.to_datetime(
            data_temp['PeriodEnd'] if 'PeriodEnd' in data_temp.columns else data_temp['period_end'])
        data_temp = data_temp.drop(
            ['PeriodStart', 'Period', 'PeriodEnd'] if 'PeriodEnd' in data_temp.columns else ['period_end', 'period'],
            axis=1)

        if 'PeriodEnd' not in data_temp.columns:
            data_temp.rename(
                columns={'ghi': 'Ghi', 'ebh': 'Ebh', 'dni': 'Dni', 'dhi': 'Dhi', 'cloud_opacity': 'CloudOpacity'},
                inplace=True)
            # data_temp=data_temp[['Ghi', 'Ebh', 'Dni', 'Dhi', 'CloudOpacity', 'Date']]
        data = pd.concat([data, data_temp], ignore_index=True)[
            ['Ghi', 'Ebh', 'Dni', 'Dhi', 'CloudOpacity', 'AirTemp', 'Date']]

    data.set_index('Date', inplace=True)
    data.index = data.index.tz_convert("Atlantic/Madeira")

    data = data.resample(rule='30T', closed='left', label='right').mean()
    data = data[~data.index.duplicated(keep='first')]
    data = data.asfreq('30min')
    data = data.sort_index()
    return data


## Load power data
def load_all_phases_data():
    power = pd.DataFrame()
    for path in ['../../Dataset/PT/substation_2018_2021.csv', '../../Dataset/PT/substation_2020_2021.csv',
                 '../../Dataset/PT/substation_all_phases.csv']:
        data_temp = pd.read_csv(path)

        if 'measure_grid' not in data_temp.columns:
            data_temp.rename(columns={'measure_cons': 'measure_grid'}, inplace=True)

        # data_temp = data_temp[['timestamp', 'measure_grid']]
        data_temp.rename(columns={'measure_grid': 'NetLoad'}, inplace=True)
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp'])

        power = pd.concat([power, data_temp], ignore_index=True)

    power.set_index('timestamp', inplace=True)
    power.index = power.index.tz_convert("Atlantic/Madeira")

    power = power.resample(rule='30T', closed='left', label='right').mean()
    power = power[~power.index.duplicated(keep='first')]
    power = power.asfreq('30min')
    power = power.sort_index()

    power['LI'] = power[['L3.I', 'L2.I', 'L1.I']].sum(1)
    power['LQ'] = power[['L3.Q', 'L2.Q', 'L1.Q']].sum(1)
    power['LP'] = power[['L3.P', 'L2.P', 'L1.P']].sum(1)
    power['LS'] = power[['L3.S', 'L2.S', 'L1.S']].sum(1)
    power['LPF'] = power[['L3.PF', 'L2.PF', 'L1.PF']].mean(1)
    power['LV'] = power[['L3.V', 'L2.V', 'L1.V']].mean(1)

    # convernt power into kW
    power[['LQ', 'LP', "LS", "NetLoad"]] = power[['LQ', 'LP', "LS", "NetLoad"]].values / 1000

    return power[['LI', 'LQ', 'LP', "LS", "LPF", "LV", "NetLoad"]]


def load_all_phases_PT(SAMPLES_PER_DAY=48, add_ghi=True):
    radiation = load_irradiance_data()['2019':'2021']
    radiation = radiation.ffill(limit=2)['2019':'2021']
    load = load_all_phases_data()
    load = load.ffill(limit=2)

    print(f"Total data sample: {load.shape[0]}")
    print(f"Missing data sample: {load.isnull().sum()[0]}")
    print(f" percentage of Missing data sample: {load.isnull().sum()[0] / len(load)}")

    data = pd.merge(radiation, load, how='outer', left_index=True, right_index=True)['2019':'2021']
    data = clean_data(data, SAMPLES_PER_DAY=SAMPLES_PER_DAY, columns=['NetLoad'])
    data.sort_index(inplace=True)

    print(" ")
    print(f"Total data sample after cleaning: {data.shape[0]}")
    print(f"Missing data sample after cleaning: {data.isnull().sum()[0]}")
    print(f" percentage of data loss : {1 - (data.shape[0] / len(load))}")

    # data.index = data.index.astype('datetime64[ns]')
    data.index.name = 'timestamp'
    min_dt = min([data.index.min()])
    max_dt = max([data.index.max()]) + pd.Timedelta(minutes=30)
    dt_rng = pd.date_range(min_dt, max_dt, freq='30T')
    data = data.reindex(dt_rng)

    data.index.name = 'timestamp'
    data = add_exogenous_variables(data.reset_index(), one_hot=False)
    data = data.set_index("timestamp")

    # remove anamoly airtime data

    ##interpolate missing values with RFregressor
    columns = ['AirTemp', 'CloudOpacity', 'Ebh', 'Dni', 'Dhi', 'Ghi', 'LI', 'LQ', 'LP', "LS", "LPF", "LV", "NetLoad"]
    features = [
        'DAYOFWEEK', 'WEEK', 'MONTH', 'HOUR']
    for column in columns:
        df_feats = data[features + [column]]

        missing_site_power_dts = data.index[data[column].isnull()]
        missing_dt_X = df_feats.loc[missing_site_power_dts].drop(column, axis=1).values
        X, y = split_X_y_data(df_feats, column)
        model = RandomForestRegressor()
        model.fit(X, y)

        data.loc[missing_site_power_dts, column] = model.predict(missing_dt_X)
        features.append(column)

    data.rename(columns={'AirTemp': 'Temperature'}, inplace=True)
    data = add_time_features(data, hemisphere='Northern')
    data['Season'] = pd.Categorical(data['Season'])
    data['Season'] = data['Season'].astype('category').cat.codes
    # data['NetLoad']=signal.medfilt(data['NetLoad'].values, 3)

    if add_ghi:
        data['NetLoad-Ghi'] = np.nan
        net_load_ghi = compute_netload_ghi(data['NetLoad'].values, data['Ghi'].values, SAMPLES_PER_DAY).flatten()
        data = data.iloc[len(data) - len(net_load_ghi):]
        data['NetLoad-Ghi'] = net_load_ghi
        del net_load_ghi

    print(" ")
    print(f"Total data sample after cleaning: {data.shape[0]}")
    print(f"Missing data sample after cleaning: {data.isnull().sum()[0]}")
    print(f" percentage of data loss : {1 - (data.shape[0] / len(load))}")

    return data


def load_all_phases_PT(SAMPLES_PER_DAY=48, add_ghi=True):
    radiation = load_irradiance_data()['2019':'2021']
    radiation = radiation.ffill(limit=2)['2019':'2021']
    load = load_all_phases_data()
    load = load.ffill(limit=2)

    print(f"Total data sample: {load.shape[0]}")
    print(f"Missing data sample: {load.isnull().sum()[0]}")
    print(f" percentage of Missing data sample: {load.isnull().sum()[0] / len(load)}")

    data = pd.merge(radiation, load, how='outer', left_index=True, right_index=True)['2019':'2021']
    data = clean_data(data, SAMPLES_PER_DAY=SAMPLES_PER_DAY, columns=['NetLoad'])
    data.sort_index(inplace=True)

    print(" ")
    print(f"Total data sample after cleaning: {data.shape[0]}")
    print(f"Missing data sample after cleaning: {data.isnull().sum()[0]}")
    print(f" percentage of data loss : {1 - (data.shape[0] / len(load))}")

    # data.index = data.index.astype('datetime64[ns]')
    data.index.name = 'timestamp'
    min_dt = min([data.index.min()])
    max_dt = max([data.index.max()]) + pd.Timedelta(minutes=30)
    dt_rng = pd.date_range(min_dt, max_dt, freq='30T')
    data = data.reindex(dt_rng)

    data.index.name = 'timestamp'
    data = add_exogenous_variables(data.reset_index(), one_hot=False)
    data = data.set_index("timestamp")

    # remove anamoly airtime data

    ##interpolate missing values with RFregressor
    columns = ['AirTemp', 'CloudOpacity', 'Ebh', 'Dni', 'Dhi', 'Ghi', 'LI', 'LQ', 'LP', "LS", "LPF", "LV", "NetLoad"]
    features = [
        'DAYOFWEEK', 'WEEK', 'MONTH', 'HOUR']
    for column in columns:
        df_feats = data[features + [column]]

        missing_site_power_dts = data.index[data[column].isnull()]
        missing_dt_X = df_feats.loc[missing_site_power_dts].drop(column, axis=1).values
        X, y = split_X_y_data(df_feats, column)
        model = RandomForestRegressor()
        model.fit(X, y)

        data.loc[missing_site_power_dts, column] = model.predict(missing_dt_X)
        features.append(column)

    data.rename(columns={'AirTemp': 'Temperature'}, inplace=True)
    data = add_time_features(data, hemisphere='Northern')
    data['Season'] = pd.Categorical(data['Season'])
    data['Season'] = data['Season'].astype('category').cat.codes
    # data['NetLoad']=signal.medfilt(data['NetLoad'].values, 3)

    if add_ghi:
        data['NetLoad-Ghi'] = np.nan
        net_load_ghi = compute_netload_ghi(data['NetLoad'].values, data['Ghi'].values, SAMPLES_PER_DAY).flatten()
        data = data.iloc[len(data) - len(net_load_ghi):]
        data['NetLoad-Ghi'] = net_load_ghi
        del net_load_ghi

    print(" ")
    print(f"Total data sample after cleaning: {data.shape[0]}")
    print(f"Missing data sample after cleaning: {data.isnull().sum()[0]}")
    print(f" percentage of data loss : {1 - (data.shape[0] / len(load))}")

    return data
