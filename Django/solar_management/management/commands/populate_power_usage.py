import csv
from django.core.management.base import BaseCommand
from solar_management.models import PowerUsage
from datetime import datetime

class Command(BaseCommand):
    help = 'Populates PowerUsage table with data'

    def handle(self, *args, **kwargs):
        # Replace this with the file path if you're reading from a file
        data = """
        16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000
        16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000
        16/12/2006;17:26:00;5.374;0.498;233.290;23.000;0.000;2.000;17.000
        16/12/2006;17:27:00;5.388;0.502;233.740;23.000;0.000;1.000;17.000
        16/12/2006;17:28:00;3.666;0.528;235.680;15.800;0.000;1.000;17.000
        16/12/2006;17:29:00;3.520;0.522;235.020;15.000;0.000;2.000;17.000
        16/12/2006;17:30:00;3.702;0.520;235.090;15.800;0.000;1.000;17.000
        16/12/2006;17:31:00;3.700;0.520;235.220;15.800;0.000;1.000;17.000
        16/12/2006;17:32:00;3.668;0.510;233.990;15.800;0.000;1.000;17.000
        16/12/2006;17:33:00;3.662;0.510;233.860;15.800;0.000;2.000;16.000
        16/12/2006;17:34:00;4.448;0.498;232.860;19.600;0.000;1.000;17.000
        16/12/2006;17:35:00;5.412;0.470;232.780;23.200;0.000;1.000;17.000
        16/12/2006;17:36:00;5.224;0.478;232.990;22.400;0.000;1.000;16.000
        16/12/2006;17:37:00;5.268;0.398;232.910;22.600;0.000;2.000;17.000
        16/12/2006;17:38:00;4.054;0.422;235.240;17.600;0.000;1.000;17.000
        16/12/2006;17:39:00;3.384;0.282;237.140;14.200;0.000;0.000;17.000
        16/12/2006;17:40:00;3.270;0.152;236.730;13.800;0.000;0.000;17.000
        16/12/2006;17:41:00;3.430;0.156;237.060;14.400;0.000;0.000;17.000
        16/12/2006;17:42:00;3.266;0.000;237.130;13.800;0.000;0.000;18.000
        16/12/2006;17:43:00;3.728;0.000;235.840;16.400;0.000;0.000;17.000
        16/12/2006;17:44:00;5.894;0.000;232.690;25.400;0.000;0.000;16.000
        16/12/2006;17:45:00;7.706;0.000;230.980;33.200;0.000;0.000;17.000
        """

        # Parse the data and save to the database
        data = data.strip().split("\n")
        for row in data:
            row_data = row.strip().split(";")
            dt_str = f"{row_data[0]} {row_data[1]}"
            dt = datetime.strptime(dt_str, '%d/%m/%Y %H:%M:%S')
            global_active_power = float(row_data[2])
            global_reactive_power = float(row_data[3])
            voltage = float(row_data[4])
            global_intensity = float(row_data[5])
            sub_metering_1 = float(row_data[6])
            sub_metering_2 = float(row_data[7])
            sub_metering_3 = float(row_data[8])

            # Create and save PowerUsage object
            PowerUsage.objects.create(
                dt=dt,
                global_active_power=global_active_power,
                global_reactive_power=global_reactive_power,
                voltage=voltage,
                global_intensity=global_intensity,
                sub_metering_1=sub_metering_1,
                sub_metering_2=sub_metering_2,
                sub_metering_3=sub_metering_3
            )

        self.stdout.write(self.style.SUCCESS('Successfully populated PowerUsage table with data.'))
