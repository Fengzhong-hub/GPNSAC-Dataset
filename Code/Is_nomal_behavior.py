"""
Determine whether the behavior is within normal authority based on the role

"""
from enum import Enum

# Defining Object Resources
class Resource(Enum):
    gas_flow = 'DB10.52'  # Network traffic
    temperature_high = 'DB10.22'  # High pressure pipe temperature
    #temperature_mid = 'DB10.32'  # Medium pressure pipeline temperature
    # pressure_in = 'DB10.2'  # High pressure pipeline inlet pressure
    # pressure_out = 'DB10.12'  # High pressure pipeline outlet pressure
    pressure_high = 'DB10.2' # High pressure pipeline pressure
    pressure_mid = 'DB10.42' # Medium pressure pipeline pressure
    pressure_low = '110' # Low pressure pipeline pressure
    valve_low = '0'  # Shut-off valve low pressure
    valve_mid = 'Q0.1'  # Cut-off valve medium pressure
    valve_high = 'Q0.0'  # Cut-off valve high pressure


# # Define roles with permissions

def role_access(role, resource):
    roles_permissions = {
        # Network traffic, temperature
        "Temp_role": [Resource.gas_flow.value, Resource.temperature_high.value],
        # Network traffic, temperature , Low pressure, cut-off valve low pressure
        "Role1": [Resource.gas_flow.value, Resource.temperature_high.value,
                  Resource.pressure_low.value, Resource.valve_low.value],
        # Network traffic, temperature, Medium pressure, cut-off valve medium pressure
        "Role2": [Resource.gas_flow.value, Resource.temperature_high.value,
                  Resource.pressure_mid.value, Resource.valve_mid.value],
        # Network traffic, temperature, High pressure, cut-off valve high pressure
        "Role3": [Resource.gas_flow.value, Resource.temperature_high.value,
                  Resource.pressure_high.value, Resource.valve_high.value]
    }
    if resource in roles_permissions[role]:
        return True
    else:
        return False


if __name__ == '__main__':
    role = 'Role2'
    #resource = '110'
    resource = 'Q0.1'
    print(role_access(role, resource))
