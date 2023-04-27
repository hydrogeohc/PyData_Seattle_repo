import json
import subprocess

def get_carbon_emissions(start_date, end_date):
    try:
        result = subprocess.run(
            [
                "Cloud_carbon",
                "AWS",
                "--startDate",
                start_date,
                "--endDate",
                end_date,
                "--groupBy",
                "day",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise Exception(f"Error: {result.stderr}")

        emissions_data = json.loads(result.stdout)
        return emissions_data

    except Exception as e:
        print(f"Error while fetching carbon emissions data: {e}")
        return None


def main():
    start_date = "2022-08-01"
    end_date = "2022-08-31"

    emissions_data = get_carbon_emissions(start_date, end_date)

    if emissions_data:
        print(f"Carbon emissions data from {start_date} to {end_date}:")
        for day in emissions_data:
            print(f"{day['timestamp']}: {day['kilowattHours']} kWh, {day['co2e']} kg CO2e")


if __name__ == "__main__":
    main()
