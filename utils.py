import pandas as pd

def extract_datetime_features(dt):
    def get_time_periods(hour):
        if 3 <= hour < 7:
            return 1
        elif 7 <= hour < 12:
            return 2
        elif 12 <= hour < 16:
            return 3
        elif 16 <= hour < 22:
            return 4
        return 0
    
    # day_quarter_mapping = {'Night': 0, 'Dawn': 1, 'Morning': 2, 'Afternoon': 3, 'Evening': 4}
    # events['Day Period'] = events['Day Period'].map(day_quarter_mapping)

    return pd.DataFrame([{
        'day_of_week': dt.weekday(),
        'Year': dt.year,
        'Month': dt.month,
        'Day': dt.day,
        'Hour': dt.hour,
        'minute': dt.minute,
        'Day Period': get_time_periods(dt.hour)
    }])

def get_item_descriptions(item_ids, description_df):
    output = []
    for item in item_ids:
        subset = description_df[description_df['itemid'] == item]
        if not subset.empty:
            start = "<br>Описание<br><br>"
            text = "<br>".join(
                subset.sort_values("property").apply(lambda x: f"<strong>{x['property']}:</strong> \t{x['value']}", axis=1)
            )
            text = start + text
        else:
            text = "Описание товара не найдено"
        output.append(f"<div><strong>Товар {item}:</strong><br>{text}</div><br>")

    return output

