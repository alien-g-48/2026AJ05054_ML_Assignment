"""Download the public UCI Phishing Websites dataset directly into memory.

The full dataset is never cached or saved in the project. Internet access is
needed only when training the bundled dataset again; saved-model evaluation
uses local pickle pipelines and the held-out test CSV.
"""

from copy import deepcopy
from http.client import HTTPException
from io import BytesIO, StringIO
import hashlib
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from scipy.io import arff

PHISHING_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
ARFF_FILENAME = "Training Dataset.arff"
TARGET_COLUMN = "is_phishing"

# Source attribution, original feature spelling, and accepted integer codes.
# Observed checksums below identify the version used for the bundled models.
SOURCE_METADATA = {'name': 'UCI Phishing Websites',
 'source_url': 'https://archive.ics.uci.edu/dataset/327/phishing+websites',
 'download_url': 'https://archive.ics.uci.edu/static/public/327/phishing+websites.zip',
 'doi': '10.24432/C51W2X',
 'license': 'CC BY 4.0',
 'citation': 'Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine '
             'Learning Repository. https://doi.org/10.24432/C51W2X.',
 'original_rows': 11055,
 'feature_count': 30,
 'archive_sha256': '514f582cb44cba17251baf1d59f64cddff16efa2b644d091ce8d2a1e8d492940',
 'arff_sha256': '3bcf93b1612306537ee2737751f1bac185b2a77886ab376644c3bf0c7f7cb3e6',
 'target_mapping': {'UCI Result -1': 'is_phishing 1 (Phishing)',
                    'UCI Result 1': 'is_phishing 0 (Legitimate)'},
 'positive_class': '1 = Phishing',
 'feature_names': ['having_IP_Address',
                   'URL_Length',
                   'Shortining_Service',
                   'having_At_Symbol',
                   'double_slash_redirecting',
                   'Prefix_Suffix',
                   'having_Sub_Domain',
                   'SSLfinal_State',
                   'Domain_registeration_length',
                   'Favicon',
                   'port',
                   'HTTPS_token',
                   'Request_URL',
                   'URL_of_Anchor',
                   'Links_in_tags',
                   'SFH',
                   'Submitting_to_email',
                   'Abnormal_URL',
                   'Redirect',
                   'on_mouseover',
                   'RightClick',
                   'popUpWidnow',
                   'Iframe',
                   'age_of_domain',
                   'DNSRecord',
                   'web_traffic',
                   'Page_Rank',
                   'Google_Index',
                   'Links_pointing_to_page',
                   'Statistical_report'],
 'feature_values': {'having_IP_Address': [-1, 1],
                    'URL_Length': [-1, 0, 1],
                    'Shortining_Service': [-1, 1],
                    'having_At_Symbol': [-1, 1],
                    'double_slash_redirecting': [-1, 1],
                    'Prefix_Suffix': [-1, 1],
                    'having_Sub_Domain': [-1, 0, 1],
                    'SSLfinal_State': [-1, 0, 1],
                    'Domain_registeration_length': [-1, 1],
                    'Favicon': [-1, 1],
                    'port': [-1, 1],
                    'HTTPS_token': [-1, 1],
                    'Request_URL': [-1, 1],
                    'URL_of_Anchor': [-1, 0, 1],
                    'Links_in_tags': [-1, 0, 1],
                    'SFH': [-1, 0, 1],
                    'Submitting_to_email': [-1, 1],
                    'Abnormal_URL': [-1, 1],
                    'Redirect': [0, 1],
                    'on_mouseover': [-1, 1],
                    'RightClick': [-1, 1],
                    'popUpWidnow': [-1, 1],
                    'Iframe': [-1, 1],
                    'age_of_domain': [-1, 1],
                    'DNSRecord': [-1, 1],
                    'web_traffic': [-1, 0, 1],
                    'Page_Rank': [-1, 1],
                    'Google_Index': [-1, 1],
                    'Links_pointing_to_page': [-1, 0, 1],
                    'Statistical_report': [-1, 1]}}


def load_phishing_dataset():
    """Return (frame, provenance) from the official archive, without disk I/O.

    Original UCI Result -1 denotes phishing and becomes is_phishing=1;
    original Result 1 denotes legitimate and becomes is_phishing=0.
    Connection failures are reported explicitly; there is no local fallback.
    """
    try:
        with urlopen(PHISHING_DOWNLOAD_URL, timeout=45) as response:
            payload = response.read()
    except (URLError, OSError, TimeoutError, HTTPException) as exc:
        raise RuntimeError(
            "Could not download the dataset from UCI. Check your internet "
            "connection and retry: " + PHISHING_DOWNLOAD_URL
        ) from exc
    try:
        with ZipFile(BytesIO(payload)) as archive:
            arff_bytes = archive.read(ARFF_FILENAME)
        records, _ = arff.loadarff(StringIO(arff_bytes.decode("utf-8-sig")))
        original = pd.DataFrame(records).apply(pd.to_numeric, errors="raise")
    except (BadZipFile, KeyError, UnicodeError, ValueError, TypeError, OSError) as exc:
        raise RuntimeError("The UCI download did not contain a valid Training Dataset.arff file.") from exc

    source = deepcopy(SOURCE_METADATA)
    features = source["feature_names"]
    if list(original.columns) != [*features, "Result"]:
        raise RuntimeError("The UCI dataset columns changed; review the source schema before training.")
    if len(original) < 500 or len(features) < 12:
        raise RuntimeError("The downloaded UCI dataset does not meet the assignment's minimum size.")
    if not original["Result"].isin([-1, 1]).all():
        raise RuntimeError("The UCI target labels changed; expected Result codes -1 and 1.")
    for name in features:
        if not original[name].isin(source["feature_values"][name]).all():
            raise RuntimeError(f"The UCI feature codes changed in {name}.")

    frame = original[features].astype(np.int8).copy()
    frame[TARGET_COLUMN] = original["Result"].map({-1: 1, 1: 0}).astype(np.int8)
    source["original_rows"] = len(frame)
    source["archive_sha256"] = hashlib.sha256(payload).hexdigest()
    source["arff_sha256"] = hashlib.sha256(arff_bytes).hexdigest()
    source["csv_sha256"] = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
    source["target_column"] = TARGET_COLUMN
    return frame, source
