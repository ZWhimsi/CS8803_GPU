TRACE_FILE=macsim_traces.tar.gz

if test -d ${MACSIM_TRACE_DIR}; then
  echo "> Traces found!  (${MACSIM_TRACE_DIR})";
else
  if [ ${MACSIM_TRACE_DIR} = "${MACSIM_DIR}/macsim_traces" ]; then
    # We're using local trace directory 
    echo "> Downloading traces...";
    curl 'https://gtvault-my.sharepoint.com/personal/ssingh849_gatech_edu/_layouts/15/download.aspx?UniqueId=2d08d7b3%2D1393%2D49be%2D84c2%2Dccd240220b14' \
    -H 'authority: gtvault-my.sharepoint.com' \
    -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.7' \
    -H 'accept-language: en-US,en;q=0.9' \
    -H $'cookie: Syntex_Translation_Languages=[["af","Afrikaans"],["am","Amharic"],["ar","Arabic"],["as","Assamese"],["az","Azerbaijani"],["ba","Bashkir"],["bg","Bulgarian"],["bho","Bhojpuri"],["bn","Bangla"],["bo","Tibetan"],["brx","Bodo"],["bs","Bosnian"],["ca","Catalan"],["cs","Czech"],["cy","Welsh"],["da","Danish"],["de","German"],["doi","Dogri"],["dsb","Lower Sorbian"],["dv","Divehi"],["el","Greek"],["en","English"],["es","Spanish"],["et","Estonian"],["eu","Basque"],["fa","Persian"],["fi","Finnish"],["fil","Filipino"],["fj","Fijian"],["fo","Faroese"],["fr","French"],["fr-CA","French (Canada)"],["ga","Irish"],["gl","Galician"],["gom","Konkani"],["gu","Gujarati"],["ha","Hausa"],["he","Hebrew"],["hi","Hindi"],["hne","Chhattisgarhi"],["hr","Croatian"],["hsb","Upper Sorbian"],["ht","Haitian Creole"],["hu","Hungarian"],["hy","Armenian"],["id","Indonesian"],["ig","Igbo"],["ikt","Inuinnaqtun"],["is","Icelandic"],["it","Italian"],["iu","Inuktitut"],["iu-Latn","Inuktitut (Latin)"],["ja","Japanese"],["ka","Georgian"],["kk","Kazakh"],["km","Khmer"],["kmr","Kurdish (Northern)"],["kn","Kannada"],["ko","Korean"],["ks","Kashmiri"],["ku","Kurdish (Central)"],["ky","Kyrgyz"],["ln","Lingala"],["lo","Lao"],["lt","Lithuanian"],["lug","Ganda"],["lv","Latvian"],["lzh","Chinese (Literary)"],["mai","Maithili"],["mg","Malagasy"],["mi","MÄ\u0081ori"],["mk","Macedonian"],["ml","Malayalam"],["mn-Cyrl","Mongolian (Cyrillic)"],["mn-Mong","Mongolian (Traditional)"],["mni","Manipuri"],["mr","Marathi"],["ms","Malay"],["mt","Maltese"],["mww","Hmong Daw"],["my","Myanmar (Burmese)"],["nb","Norwegian"],["ne","Nepali"],["nl","Dutch"],["nso","Sesotho sa Leboa"],["nya","Nyanja"],["or","Odia"],["otq","QuerÃ©taro Otomi"],["pa","Punjabi"],["pl","Polish"],["prs","Dari"],["ps","Pashto"],["pt","Portuguese (Brazil)"],["pt-PT","Portuguese (Portugal)"],["ro","Romanian"],["ru","Russian"],["run","Rundi"],["rw","Kinyarwanda"],["sd","Sindhi"],["si","Sinhala"],["sk","Slovak"],["sl","Slovenian"],["sm","Samoan"],["sn","Shona"],["so","Somali"],["sq","Albanian"],["sr-Cyrl","Serbian (Cyrillic)"],["sr-Latn","Serbian (Latin)"],["st","Sesotho"],["sv","Swedish"],["sw","Swahili"],["ta","Tamil"],["te","Telugu"],["th","Thai"],["ti","Tigrinya"],["tk","Turkmen"],["tlh-Latn","Klingon (Latin)"],["tlh-Piqd","Klingon (pIqaD)"],["tn","Setswana"],["to","Tongan"],["tr","Turkish"],["tt","Tatar"],["ty","Tahitian"],["ug","Uyghur"],["uk","Ukrainian"],["ur","Urdu"],["uz","Uzbek (Latin)"],["vi","Vietnamese"],["xh","Xhosa"],["yo","Yoruba"],["yua","Yucatec Maya"],["yue","Cantonese (Traditional)"],["zh-Hans","Chinese Simplified"],["zh-Hant","Chinese Traditional"],["zu","Zulu"]]; MSFPC=GUID=e2699baaa800470085a994721787e2a3&HASH=e269&LV=202308&V=4&LU=1693178173855; MicrosoftApplicationsTelemetryDeviceId=f9865078-e914-473d-8db5-9eb0fd52621b; VisioWacDataCenter=PUS9; SIMI=eyJzdCI6MH0=; OneNoteWacDataCenter=PUS13; ExcelWacDataCenter=PUS13; WordWacDataCenter=PUS7; rtFa=Cmewh+LHu74IiPHaLUw0A2eHIoFgJ0oGrwZ/VKUIPXkmYmZjYTUwYTYtMzliZS00ZTRjLWJjMGQtMzY3MzQ5ZjcyNWIwIzEzMzUzNDQ5NDM5NDYwNzA2NiMyYjkxMGZhMS02MDBhLTQwMDAtZWUyMS0xODA1YTk1ZGMzNGIjc3NpbmdoODQ5JTQwZ2F0ZWNoLmVkdSMxOTMwNTQjdnRHQ2JlVEdQZGlzUUtlaHZFc1BGLTFhTUpJl1xmuD2+S3+ieXITYv9JO4RhImL+Su8yFFm6IUxxSjgRqGi3OiJeUW0r05iSzclAifVIVvKHLDjojUXUP5ILr6+Z2zVQQM28k3SJ7nQRolOPyr6VeTC9OxuVcG4LYlwQQuW9YgK345Y0OtQ2CM9e4KgXHqSTqjEmZbGrpmb9PoQ+WoXhxY6LNa22UX8ddtnMujHjbzP+WWZeAClW6kFS1CFZexF7OX5LDB4lt6bzNd/YQPYxslkDLU0Vy7c3jnmd5ufs6HxG/bnGvcJZaLjIPuGkNucuFlsXRGlBCeNNNTlWojIejZDGzCGQ0vS1f3ngPHSNdzmBguhgLfXiAuD9a7cAAAA=; PowerPointWacDataCenter=PUS3; WacDataCenter=PUS3; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHwxMDAzMjAwMmI3MTFjMjgzQGxpdmUuY29tLDAjLmZ8bWVtYmVyc2hpcHxzc2luZ2g4NDlAZ2F0ZWNoLmVkdSwxMzM0NjQ0MTAzNDAwMDAwMDAsMTMzMjkwNjU2NzYwMDAwMDAwLDEzMzU0MzE4NzgzNzQ0NjI1MiwyNjEwOjE0ODoyMDViOjA6ZDgxNzpjOTFiOjIxMGE6ZDVlZSwzLGJmY2E1MGE2LTM5YmUtNGU0Yy1iYzBkLTM2NzM0OWY3MjViMCwsNTYzYTY0MzItMDAyMi00ZDVkLTliOTYtNTZlMzRjYWI4YThjLDQ1YmEwOWExLTYwMGItNDAwMC1kODFlLWU2MjNkOTM3ZmJmNyw0MDMyMTFhMS0wMDhmLTQwMDAtZmY5Yi02MjA0MTJjYTY3MWYsLDAsMTMzNTM5NjE4NTg0NjAxODU0LDEzMzU0MTM0NjU4NDYwMTg1NCwsLGV5SjRiWE5mWTJNaU9pSmJYQ0pEVURGY0lsMGlMQ0o0YlhOZmMzTnRJam9pTVNJc0luQnlaV1psY25KbFpGOTFjMlZ5Ym1GdFpTSTZJbk56YVc1bmFEZzBPVUJuWVhSbFkyZ3VaV1IxSWl3aWRYUnBJam9pVXkxZlpqWkhXa1U0TUdWV1FuUlpiRXRDVFhWQlFTSjksMjY1MDQ2Nzc0Mzk5OTk5OTk5OSwxMzM1MTg4MTkyNDAwMDAwMDAsZDQ0Y2IxNjYtM2JmYy00MWMwLTk0NGEtZTA4N2QyOTM4MDgzLCwsLCwsMCwsMTkzMDU0LERhREFmalFRbXB5T1h4MlJyS19XNWx2b0xaNCxwMEcwemNiMWRmQXR1QzUyMGJyZVNiMHFaU0h1bnNZdlJvb2t4UE0xcjI4bVZsa0Y2WHUzVnZSQ212d3U1aDBEQmN2SnpvWW9SMHh1S0JLeXc5N2lZZWdsZXl3c2RGQlBtaXZJMUNYeDFkZ1JhcWpSRm9CSWVoMzJvVjRaQlllZFZMU2YwL3VmU2F1SmlHOTYxLzIzcTVDMDBiSXRxQXJYK3NEZ01NMEZ0MlQ2S2cyNlVGTngvSzZta0wwcXU0Ymc2SWtta01pNm1ES3BSNU5pNnY1bnRrQU1nMzJjUHNxK0pSODgraHcxR1A1ZnRwSUNQb0VFaElBdGFmVGg2TEVMdlZtcDhWY2dWemRMTlB5VCtsZlVBbkVacHQ0czQ1Y09iWXJ0QTROVzdWM0wvdTVuYVlPcUJNaks3ZGd2YTB0NStxYjlBWjFNV2NvY1o5NHJ6NU02ZXc9PTwvU1A+; WSS_FullScreenMode=false; odbn=1; ai_session=GLoTFWfK5eUnu7QChl/DHO|1709413185741|1709413904448' \
    -H 'referer: https://gtvault-my.sharepoint.com/personal/ssingh849_gatech_edu/_layouts/15/onedrive.aspx?login_hint=ssingh849%40gatech%2Eedu&id=%2Fpersonal%2Fssingh849%5Fgatech%5Fedu%2FDocuments%2FSpr24%2FOMSCS%5FCS8803%5FProj3&view=0' \
    -H 'sec-ch-ua: "Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'sec-ch-ua-platform: "Windows"' \
    -H 'sec-fetch-dest: iframe' \
    -H 'sec-fetch-mode: navigate' \
    -H 'sec-fetch-site: same-origin' \
    -H 'sec-fetch-user: ?1' \
    -H 'service-worker-navigation-preload: true' \
    -H 'upgrade-insecure-requests: 1' \
    -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36' \
    --compressed --output ${TRACE_FILE}

    if [ $? -eq 0 ]; then
      echo "> Download ${TRACE_FILE}: OK"
    else
      echo "> Download ${TRACE_FILE}: FAILED"
      exit 1
    fi

    # Extract traces
    tar -xzf ${TRACE_FILE};
    rc=$?
    rm -f ${TRACE_FILE};

    if [ ${rc} -eq 0 ]; then
      echo "> Extracting ${TRACE_FILE}: OK"
    else
      echo "> Extracting ${TRACE_FILE}: FAILED"
      exit 1
    fi
  
  else
    echo "ERROR: Trace directory not setup properly"
    exit 1
  fi
fi
