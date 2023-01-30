# def file_read(fpt1, fgt2):
#
#     IMG = 'img'
#     LABEL = 'label'
#
#     subject_dict = {
#         IMG: fpt1,
#         LABEL: fgt2,
#     }
#     print('Dataset size:', len(fpt1), 'subjects')
#
#     return subject_dict


def file_read(fpt):

    IMG = 'img'

    subject_dict = {
        IMG: fpt,
    }
    print('Dataset size:', len(subject_dict), 'subjects')

    return subject_dict