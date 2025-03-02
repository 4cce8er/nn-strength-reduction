import logging
import coloredlogs

FIELD_STYLES = dict(
    asctime=dict(color='green'),
    hostname=dict(color='magenta'),
    levelname=dict(color='black', bold=True),
    name=dict(color='blue'),
    programname=dict(color='cyan'),
    username=dict(color='yellow'),
    filename=dict(color='blue')
)
DATE_FORMAT = '%H:%M:%S'
LOG_FORMAT = '%(asctime)s %(hostname)s %(filename)s:%(lineno)d[%(process)d] %(levelname)s %(message)s'

coloredlogs.install(fmt=LOG_FORMAT, datefmt=DATE_FORMAT, field_styles=FIELD_STYLES)  # install a handler on the root logger