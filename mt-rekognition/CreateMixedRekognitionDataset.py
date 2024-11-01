from RekognitionDatasetUpload import RekognitionDatasetCreator

if __name__ == "__main__":

    project_arn = "arn:aws:rekognition:eu-west-1:825477577700:project/MineTilbud_Mix/1730380102598"

    creator = RekognitionDatasetCreator('ste-power', 'eu-west-1')

    test_manifest = []
    train_manifest = []

    with open('rekognition_manifest.jsonl', 'r') as fr:
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

    # train_dataset_arn = creator.create_dataset(project_arn, "TRAIN")
    # print(train_dataset_arn)
    creator.upload_annotations('arn:aws:rekognition:eu-west-1:825477577700:project/MineTilbud_Mix/dataset/train/1730383777846', "".join(train_manifest))
    #
    # test_dataset_arn = creator.create_dataset(project_arn, "TEST")
    # creator.upload_annotations(test_dataset_arn, "".join(test_manifest))
