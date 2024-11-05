from RekognitionDatasetUpload import RekognitionDatasetCreator

if __name__ == "__main__":

    project_arn = "arn:aws:rekognition:eu-west-1:825477577700:project/MineTilbud_mt-input2/1730465646889"

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
    print("".join(train_manifest))

    train_dataset_arn = creator.create_dataset(project_arn, "TRAIN")
    print(train_dataset_arn)
#     creator.upload_annotations('arn:aws:rekognition:eu-west-1:825477577700:project/MineTilbud_Mix/dataset/train/1730453270608', "".join(train_manifest))
#     test_string = '{"source-ref": "s3://minetilbud-example-1/mixed/2024_uge16_biltema_sommerklarhave_1.jpg", "bounding-box": {"image_size": [{"width": 1199, "height": 1542, "depth": 3}], "annotations": [{"class_id": 1, "top": 472, "left": 25, "width": 163, "height": 607}, {"class_id": 1, "top": 470, "left": 176, "width": 318, "height": 609}, {"class_id": 1, "top": 472, "left": 329, "width": 470, "height": 608}, {"class_id": 1, "top": 129, "left": 24, "width": 467, "height": 459}]}, "bounding-box-metadata": {"objects": [{"confidence": 1}, {"confidence": 1}, {"confidence": 1}, {"confidence": 1}], "class-map": {"1": "offer"}, "type": "groundtruth/object-detection", "human-annotated": "yes", "creation-date": "2024-11-01T06:30:15", "job-name": "converted-from-coco"}}\n{"source-ref": "s3://minetilbud-example-1/mixed/2024_uge16_biltema_sommerklarhave_2.jpg", "bounding-box": {"image_size": [{"width": 1199, "height": 1542, "depth": 3}], "annotations": [{"class_id": 1, "top": 352, "left": 18, "width": 123, "height": 452}, {"class_id": 1, "top": 128, "left": 17, "width": 237, "height": 341}, {"class_id": 1, "top": 348, "left": 134, "width": 236, "height": 450}, {"class_id": 1, "top": 240, "left": 244, "width": 345, "height": 452}, {"class_id": 1, "top": 15, "left": 244, "width": 347, "height": 229}, {"class_id": 1, "top": 17, "left": 17, "width": 230, "height": 119}]}, "bounding-box-metadata": {"objects": [{"confidence": 1}, {"confidence": 1}, {"confidence": 1}, {"confidence": 1}, {"confidence": 1}, {"confidence": 1}], "class-map": {"1": "offer"}, "type": "groundtruth/object-detection", "human-annotated": "yes", "creation-date": "2024-11-01T06:30:15", "job-name": "converted-from-coco"}}'
#     creator.upload_annotations('arn:aws:rekognition:eu-west-1:825477577700:project/MineTilbud_Mix/dataset/train/1730453270608', test_string)

    #
    # test_dataset_arn = creator.create_dataset(project_arn, "TEST")
    # creator.upload_annotations(test_dataset_arn, "".join(test_manifest))
