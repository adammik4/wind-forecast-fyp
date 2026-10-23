"""Download the public data behind the state-by-state analysis. Already-present files are skipped.

    python fetch_data.py

  Data/sites/<site>.csv.gz      NASA POWER hourly weather (MERRA-2) for each state's representative site,
                                same parameters, period and format as Data/Mersing.csv
  Data/malaysia_states.geojson  state boundaries (geoBoundaries gbOpen MYS ADM1, simplified, ODbL)
"""
import gzip
import json
import urllib.request

import common as c

NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=WS10M,WS50M,WD10M,T2M,PS,RH2M"
    "&community=RE&longitude={lon}&latitude={lat}&start=20150101&end=20260630&format=CSV&time-standard=LST"
)
BOUNDARIES_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/MYS/ADM1/"
                  "geoBoundaries-MYS-ADM1_simplified.geojson")
# geoBoundaries names -> the state names used in common.STATES
RENAME = {"Malacca": "Melaka", "Penang": "Pulau Pinang"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def round_coords(x):
    return [round_coords(v) for v in x] if isinstance(x[0], list) else [round(v, 3) for v in x]


def rewind(geometry: dict) -> dict:
    """Plotly's geo maps (d3-geo) want clockwise outer rings and anticlockwise holes, the reverse of the GeoJSON
    standard; with the wrong winding a state is drawn as the whole globe minus the state."""
    def area(ring):  # shoelace, positive when anticlockwise
        return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:]))

    def fix(polygon):
        return [ring if (area(ring) < 0) == (i == 0) else ring[::-1] for i, ring in enumerate(polygon)]

    coords = geometry["coordinates"]
    coords = [fix(p) for p in coords] if geometry["type"] == "MultiPolygon" else fix(coords)
    return dict(type=geometry["type"], coordinates=coords)


def main():
    for state, s in c.STATES.items():
        path = c.site_file(s["site"])
        if path.exists():
            continue
        print(f"{state}: {s['site']} ({s['lat']}, {s['lon']})")
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            fh.write(get(NASA_URL.format(lat=s["lat"], lon=s["lon"])))

    if not c.STATES_GEOJSON.exists():
        print("State boundaries")
        g = json.loads(get(BOUNDARIES_URL))
        features = [dict(type="Feature", id=RENAME.get(n := f["properties"]["shapeName"], n),
                         properties=dict(name=RENAME.get(n, n)),
                         geometry=rewind(dict(type=f["geometry"]["type"], coordinates=round_coords(f["geometry"]["coordinates"]))))
                    for f in g["features"]]
        c.STATES_GEOJSON.write_text(json.dumps(dict(type="FeatureCollection", features=features), separators=(",", ":")))


if __name__ == "__main__":
    main()
