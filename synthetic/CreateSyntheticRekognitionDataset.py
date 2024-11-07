from RekognitionDatasetUpload import RekognitionDatasetCreator

if __name__ == "__main__":

    project_arn = "arn:aws:rekognition:eu-west-1:825477577700:project/MineTilbud_Synthetic/1730809089007"

    creator = RekognitionDatasetCreator('ste-power', 'eu-west-1')

    test_manifest = []
    train_manifest = []

    with open('ad_dataset_3/custom_labels.manifest', 'r') as fr:
        i = 0
        while line := fr.readline():
            if i % 5 == 0:
                test_manifest.append(line)
            else:
                train_manifest.append(line)

            i += 1

    print(f'Size of TRAIN set {len(train_manifest)}')
    print(f'Size of TEST set  {len(test_manifest)}')

    # print("".join(test_manifest))
    # print("".join(train_manifest))

    # train_dataset_arn = creator.create_dataset(project_arn, "TRAIN")
    creator.upload_annotations('arn:aws:rekognition:eu-west-1:825477577700:project/MineTilbud_Synthetic/dataset/train/1730983156260', "".join(train_manifest))

    #
    # test_dataset_arn = creator.create_dataset(project_arn, "TEST")
    creator.upload_annotations('arn:aws:rekognition:eu-west-1:825477577700:project/MineTilbud_Synthetic/dataset/test/1730983210990', "".join(test_manifest))
