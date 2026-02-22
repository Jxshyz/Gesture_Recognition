import joblib

data = joblib.load("data/Gestures_Joschua_1.pkl")

print("TYPE:", type(data))

if isinstance(data, dict):
    print("DICT KEYS:", data.keys())

elif isinstance(data, list):
    print("LIST LENGTH:", len(data))
    print("FIRST ELEMENT TYPE:", type(data[0]))
    print("FIRST ELEMENT:", data[0])

else:
    print("CONTENT:", data)

print(data.columns.values)
