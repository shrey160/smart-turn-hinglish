# Listening Log — Day-1 audit (masterplan P0)

Sampled with seed 42 by `scripts/sample_listening_log.py`. Annotate the
`notes` column after listening: naturalness, pure-Hindi vs code-mixed,
filler realism (midfiller/endfiller), background noise, truncation artifacts.

| # | clip | lang | label | midfiller | endfiller | dur_s | rms | peak | tail_sil_s | notes |
|---|------|------|-------|-----------|-----------|-------|-----|------|------------|-------|
| 1 | `data\train_pool\hin\complete-False-False\0f91bab0-0def-44e7-b353-5a0cf8b8d7f2.flac` | hin | complete | False | False | 5.92 | 0.1226 | 1.00 | 0.00 |  |
| 2 | `data\train_pool\hin\incomplete-True-True\73f2294a-86db-484e-ae08-7509d2a8df57.flac` | hin | incomplete | True | True | 7.40 | 0.1230 | 0.77 | 0.26 |  |
| 3 | `data\train_pool\hin\incomplete-False-True\286c7f05-254e-4530-ad52-8e26b5263ea8.flac` | hin | incomplete | False | True | 5.12 | 0.1052 | 0.67 | 0.00 |  |
| 4 | `data\train_pool\hin\complete-True-False\576a7988-fd50-44cc-b1af-c7dffbc347f6.flac` | hin | complete | True | False | 5.80 | 0.1115 | 0.76 | 0.28 |  |
| 5 | `data\train_pool\hin\incomplete-False-True\61b3e6af-6c52-41b1-984a-6ea22f06fb53.flac` | hin | incomplete | False | True | 5.04 | 0.1074 | 0.87 | 0.02 |  |
| 6 | `data\train_pool\hin\incomplete-False-True\0a4acac8-8346-4402-a69b-252cddf9f388.flac` | hin | incomplete | False | True | 4.32 | 0.1264 | 0.97 | 0.00 |  |
| 7 | `data\train_pool\hin\incomplete-False-True\6930585c-2463-4b2f-a707-e921090ea8bd.flac` | hin | incomplete | False | True | 4.80 | 0.0871 | 0.61 | 0.28 |  |
| 8 | `data\train_pool\hin\complete-False-False\3246b1fb-2057-4702-b29c-7e5980645c8a.flac` | hin | complete | False | False | 7.36 | 0.0708 | 0.69 | 0.24 |  |
| 9 | `data\train_pool\hin\incomplete-True-True\13f6a0a6-e5ca-427c-b4fb-9908ffbaee86.flac` | hin | incomplete | True | True | 12.64 | 0.1240 | 0.91 | 0.00 |  |
| 10 | `data\train_pool\hin\incomplete-True-True\3763e2ba-2c31-4d53-b8db-245f56d87b4e.flac` | hin | incomplete | True | True | 7.16 | 0.0976 | 0.96 | 0.00 |  |
| 11 | `data\train_pool\hin\complete-True-False\4adba9ba-3f2d-4211-b088-8f73aa95c44b.flac` | hin | complete | True | False | 3.84 | 0.1351 | 1.00 | 0.00 |  |
| 12 | `data\train_pool\hin\complete-True-False\28e91fa3-ae94-4db2-b584-7e5626bbcf5e.flac` | hin | complete | True | False | 4.68 | 0.1188 | 0.80 | 0.16 |  |
| 13 | `data\train_pool\hin\incomplete-True-False\4dc87c05-bb8a-4e52-b297-e5abe439976b.flac` | hin | incomplete | True | False | 4.64 | 0.1055 | 0.78 | 0.00 |  |
| 14 | `data\train_pool\hin\complete-False-False\3746625e-e92b-41eb-a33a-f39b162b7451.flac` | hin | complete | False | False | 6.80 | 0.1293 | 0.73 | 0.16 |  |
| 15 | `data\train_pool\hin\incomplete-True-False\5ce9261f-f7f1-44db-829e-5938f9b3b117.flac` | hin | incomplete | True | False | 4.52 | 0.1314 | 0.75 | 0.12 |  |
| 16 | `data\train_pool\hin\complete-False-False\0d6ce609-88c4-47e2-b0f1-d2cbe3615f37.flac` | hin | complete | False | False | 9.24 | 0.1111 | 0.99 | 0.00 |  |
| 17 | `data\train_pool\hin\complete-True-False\007979fa-f0aa-4491-b9a8-a8c7f554cd65.flac` | hin | complete | True | False | 7.24 | 0.1069 | 0.86 | 0.04 |  |
| 18 | `data\train_pool\hin\incomplete-True-True\207d34ad-3e72-4c92-a06c-19c1d5d7aab0.flac` | hin | incomplete | True | True | 10.96 | 0.0885 | 0.91 | 0.00 |  |
| 19 | `data\train_pool\hin\complete-False-False\43dd636f-f079-48dd-a524-36453f06f2f8.flac` | hin | complete | False | False | 4.36 | 0.1366 | 1.00 | 0.28 |  |
| 20 | `data\train_pool\hin\complete-False-False\764eab85-225a-4ce9-b9a7-4fbf0de89631.flac` | hin | complete | False | False | 8.56 | 0.0650 | 0.52 | 0.02 |  |
| 21 | `data\train_pool\hin\incomplete-False-True\626069f2-9587-44c4-aa22-ff30ab50771a.flac` | hin | incomplete | False | True | 9.40 | 0.0729 | 0.94 | 0.00 |  |
| 22 | `data\train_pool\hin\complete-False-False\2b1e9888-e40a-481a-8386-dff42b99e3ee.flac` | hin | complete | False | False | 6.28 | 0.1152 | 0.76 | 0.00 |  |
| 23 | `data\train_pool\hin\complete-False-False\11022118-92aa-49fa-b3e2-5103cd3d9c6a.flac` | hin | complete | False | False | 7.20 | 0.0696 | 0.51 | 0.00 |  |
| 24 | `data\train_pool\hin\incomplete-False-True\5800e8a7-1b38-4c8a-8758-e19790f62282.flac` | hin | incomplete | False | True | 2.48 | 0.0780 | 0.50 | 0.00 |  |
| 25 | `data\train_pool\hin\complete-False-False\6c15a589-bee9-4c79-8b4c-d9187770bf6c.flac` | hin | complete | False | False | 10.72 | 0.0886 | 1.00 | 0.20 |  |
| 26 | `data\train_pool\hin\incomplete-True-True\4fdc1cae-032e-4799-8281-c51c090fee7f.flac` | hin | incomplete | True | True | 10.00 | 0.1256 | 0.85 | 0.26 |  |
| 27 | `data\train_pool\hin\complete-True-False\65e479c9-be03-4a4b-a15c-99776c1964ad.flac` | hin | complete | True | False | 7.56 | 0.1337 | 0.98 | 0.00 |  |
| 28 | `data\train_pool\hin\incomplete-True-True\6ac911fd-92db-4b8c-a515-46bea9b4fa0e.flac` | hin | incomplete | True | True | 3.92 | 0.0714 | 0.64 | 0.00 |  |
| 29 | `data\train_pool\hin\complete-True-False\71567281-bc47-413e-b827-c8e5ae47e36f.flac` | hin | complete | True | False | 4.12 | 0.0825 | 0.75 | 0.00 |  |
| 30 | `data\train_pool\hin\incomplete-True-False\7a48a425-b54e-4af3-8e3d-d55dd2d22e12.flac` | hin | incomplete | True | False | 3.52 | 0.1194 | 1.00 | 0.16 |  |
| 31 | `data\train_pool\eng\complete-False-False\07c48b2b-7955-464d-9312-c4dc5738b880.flac` | eng | complete | False | False | 5.73 | 0.1369 | 0.84 | 0.98 |  |
| 32 | `data\train_pool\eng\incomplete-False-False\022c22a3-24be-45ac-a2b2-a59decfcfa41.flac` | eng | incomplete | False | False | 13.70 | 0.1229 | 0.84 | 0.92 |  |
| 33 | `data\train_pool\eng\incomplete-False-False\008cbb35-a3e0-4431-978e-ce1829a98eea.flac` | eng | incomplete | False | False | 5.63 | 0.0803 | 0.64 | 0.00 |  |
| 34 | `data\train_pool\eng\incomplete-False-False\005e898b-1a84-4989-805a-3e0e0684a407.flac` | eng | incomplete | False | False | 3.05 | 0.0303 | 0.23 | 0.00 |  |
| 35 | `data\train_pool\eng\incomplete-False-False\06f64ef3-878b-4171-8520-905f2d5615bb.flac` | eng | incomplete | False | False | 11.38 | 0.1373 | 0.85 | 0.44 |  |
| 36 | `data\train_pool\eng\complete-False-False\05f3a770-7d26-4c23-961b-02e18b89813a.flac` | eng | complete | False | False | 4.76 | 0.0404 | 0.41 | 0.00 |  |
| 37 | `data\train_pool\eng\incomplete-False-False\068cc893-86a9-437f-92f9-2066d53efaba.flac` | eng | incomplete | False | False | 13.08 | 0.1250 | 0.86 | 0.00 |  |
| 38 | `data\train_pool\eng\complete-False-False\033a9179-69e2-41fb-a99d-4bca9e60f779.flac` | eng | complete | False | False | 3.29 | 0.0792 | 0.52 | 0.00 |  |
| 39 | `data\train_pool\eng\complete-False-False\019bd47f-d282-4920-9e67-b6d3c2d1c3df.flac` | eng | complete | False | False | 12.32 | 0.0774 | 0.74 | 0.08 |  |
| 40 | `data\train_pool\eng\complete-True-False\0194ea03-2e5a-4b5a-bca6-34be802187c5.flac` | eng | complete | True | False | 6.00 | 0.0616 | 0.42 | 0.00 |  |
| 41 | `data\train_pool\eng\complete-False-False\058b18a7-1d83-4f33-bb17-2935be067847.flac` | eng | complete | False | False | 12.00 | 0.1435 | 0.90 | 0.00 |  |
| 42 | `data\train_pool\eng\incomplete-False-False\02b911fb-2c7b-4e9a-b99a-5e84dee132ad.flac` | eng | incomplete | False | False | 4.95 | 0.1005 | 1.00 | 0.00 |  |
| 43 | `data\train_pool\eng\complete-True-False\02fc693a-0cc6-461c-b775-824b118faabd.flac` | eng | complete | True | False | 7.88 | 0.0800 | 0.79 | 0.00 |  |
| 44 | `data\train_pool\eng\incomplete-False-False\07903ada-e61a-4098-bdca-14bb67aea458.flac` | eng | incomplete | False | False | 11.44 | 0.1236 | 0.84 | 0.96 |  |
| 45 | `data\train_pool\eng\complete-False-False\0606cf44-d76e-4649-a37d-7d623f8dc000.flac` | eng | complete | False | False | 4.39 | 0.0738 | 0.99 | 0.00 |  |
| 46 | `data\train_pool\mar\complete-True-False\0736d666-1c4d-4a50-806b-fcd4ce7d2852.flac` | mar | complete | True | False | 6.28 | 0.1292 | 0.75 | 0.24 |  |
| 47 | `data\train_pool\mar\incomplete-False-False\06eb0945-bc14-479f-b67f-946d52a6ffb3.flac` | mar | incomplete | False | False | 7.12 | 0.1344 | 0.90 | 0.06 |  |
| 48 | `data\train_pool\mar\complete-True-False\0a347766-bc29-4e6c-bb28-19a4f2ec137e.flac` | mar | complete | True | False | 6.00 | 0.1195 | 1.00 | 0.26 |  |
| 49 | `data\train_pool\mar\incomplete-True-False\0df59db0-a458-499a-a168-851d1d28023e.flac` | mar | incomplete | True | False | 3.92 | 0.1264 | 0.73 | 0.00 |  |
| 50 | `data\train_pool\mar\complete-True-False\096b4248-7eba-45f2-8c8b-ee7d289f25f5.flac` | mar | complete | True | False | 7.72 | 0.1535 | 0.98 | 0.00 |  |
| 51 | `data\train_pool\ben\incomplete-True-False\070a0679-9fe5-4f47-a659-ef792f71ca5c.flac` | ben | incomplete | True | False | 4.24 | 0.1167 | 0.75 | 0.00 |  |
| 52 | `data\train_pool\ben\incomplete-True-True\0482f8a4-890b-426d-8f39-416206bdfd85.flac` | ben | incomplete | True | True | 16.00 | 0.1014 | 0.69 | 0.22 |  |
| 53 | `data\train_pool\ben\complete-False-False\0913d3b2-051a-4647-affb-7abab5852c34.flac` | ben | complete | False | False | 10.28 | 0.0856 | 0.87 | 0.20 |  |
| 54 | `data\train_pool\ben\complete-True-False\060f3f1d-7f92-4411-9f7d-1f171f396d27.flac` | ben | complete | True | False | 10.16 | 0.1051 | 0.77 | 0.22 |  |
| 55 | `data\train_pool\ben\complete-False-False\06e5d975-acdb-48d4-8b58-e0a1f0ee8923.flac` | ben | complete | False | False | 8.00 | 0.1015 | 0.83 | 0.00 |  |
| 56 | `data\example_hinglish_fixed\train_pool\hinglish\complete-False-False\e55afc44-c9e2-5f3f-86c3-d7d41850f4fe.flac` | tts-hinglish | complete | False | False | 7.92 | 0.0598 | 0.43 | 0.16 |  |
| 57 | `data\example_hinglish_fixed\train_pool\hinglish\complete-False-False\0fb9afaa-61f7-55fe-a7e4-ca41620d643d.flac` | tts-hinglish | complete | False | False | 8.68 | 0.0699 | 0.69 | 0.20 |  |
| 58 | `data\example_hinglish_fixed\train_pool\hinglish\complete-False-False\de75a83d-fd2a-5d8b-8d8c-055bcc663965.flac` | tts-hinglish | complete | False | False | 4.07 | 0.1103 | 0.70 | 0.12 |  |
| 59 | `data\example_hinglish_fixed\train_pool\hinglish\incomplete-False-True\868f76da-2c83-5ae5-a4d6-7525053b6442.flac` | tts-hinglish | incomplete | False | True | 3.17 | 0.0952 | 0.96 | 0.06 |  |
| 60 | `data\example_hinglish_fixed\train_pool\hinglish\incomplete-False-True\b8675a71-c642-5069-bd1d-e4bda807dd6f.flac` | tts-hinglish | incomplete | False | True | 3.31 | 0.0752 | 0.59 | 0.12 |  |
