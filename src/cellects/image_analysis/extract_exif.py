#!/usr/bin/env python3
"""ADD DETAIL OF THE MODULE"""

"""

dir(my_image)
['<unknown EXIF tag 59932>', '<unknown EXIF tag 59933>', '_exif_ifd_pointer', '_gps_ifd_pointer', '_segments', 'aperture
_value', 'brightness_value', 'color_space', 'components_configuration', 'compression', 'datetime', 'datetime_digitized',
'datetime_original', 'exif_version', 'exposure_bias_value', 'exposure_mode', 'exposure_program', 'exposure_time', 'f_
number', 'flash', 'flashpix_version', 'focal_length', 'focal_length_in_35mm_film', 'get', 'get_file', 'get_thumbnail',
'gps_altitude', 'gps_altitude_ref', 'gps_datestamp', 'gps_dest_bearing', 'gps_dest_bearing_ref', 'gps_horizontal_
positioning_error', 'gps_img_direction', 'gps_img_direction_ref', 'gps_latitude', 'gps_latitude_ref', 'gps_longitude',
'gps_longitude_ref', 'gps_speed', 'gps_speed_ref', 'gps_timestamp', 'has_exif', 'jpeg_interchange_format', 'jpeg_
interchange_format_length', 'lens_make', 'lens_model', 'lens_specification', 'make', 'maker_note', 'metering_mode',
'model', 'orientation', 'photographic_sensitivity', 'pixel_x_dimension', 'pixel_y_dimension', 'resolution_unit',
'scene_capture_type', 'scene_type', 'sensing_method', 'shutter_speed_value', 'software', 'subject_area', 'subsec_time_
digitized', 'subsec_time_original', 'white_balance', 'x_resolution', 'y_and_c_positioning', 'y_resolution']

"""
import numpy as np

def extract_time(image_list, pathway="", raw_images=False):
    from numpy import min, max, any, zeros, arange, all, int64, repeat
    nb = len(image_list)
    timings = np.zeros((nb, 6), dtype=np.int64)
    if raw_images:
        import exifread
        for i in np.arange(nb):
            with open(pathway + image_list[i], 'rb') as image_file:
                my_image = exifread.process_file(image_file, details=False, stop_tag='DateTimeOriginal')
                datetime = my_image["EXIF DateTimeOriginal"]
            datetime = datetime.values[:10] + ':' + datetime.values[11:]
            timings[i, :] = datetime.split(':')
    else:
        from exif import Image
        for i in np.arange(nb):
            with open(pathway + image_list[i], 'rb') as image_file:
                my_image = Image(image_file)
                if my_image.has_exif:
                    datetime = my_image.datetime
                    datetime = datetime[:10] + ':' + datetime[11:]
                    timings[i, :] = datetime.split(':')

    if np.all(timings[:, 0] == timings[0, 0]):
        if np.all(timings[:, 1] == timings[0, 1]):
            if np.all(timings[:, 2] == timings[0, 2]):
                time = timings[:, 3] * 3600 + timings[:, 4] * 60 + timings[:, 5]
            else:
                time = timings[:, 2] * 86400 + timings[:, 3] * 3600 + timings[:, 4] * 60 + timings[:, 5]
        else:
            days_per_month = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
            for j in np.arange(nb):
                month_number = timings[j, 1]#int(timings[j, 1])
                timings[j, 1] = days_per_month[month_number] * month_number
            time = (timings[:, 1] + timings[:, 2]) * 86400 + timings[:, 3] * 3600 + timings[:, 4] * 60 + timings[:, 5]
        #time = int(time)
    else:
        time = np.repeat(0, nb)#arange(1, nb * 60, 60)#"Do not experiment the 31th of december!!!"
    if time.sum() == 0:
        time = np.repeat(0, nb)#arange(1, nb * 60, 60)
    return time
