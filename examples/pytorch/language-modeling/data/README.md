This folder contains the training data for the base architecture.
To start, the original wikipedia txt files must be uploaded in the "original" folder.
After fllowing the instructions in the language-modeling README, the ready-to-use training corpora will be available in the "final_corpora folder".

NOTE: Depending on your reseources, you may need to split large wikipedia text files (English, Russian, Arabic) and perform preprocessing in parallel on chunks.
The "create_oversampled_wikicorpus" methods support multi-processing for similar cases.