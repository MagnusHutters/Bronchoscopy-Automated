import time

class Timer:
    # Class variable to store checkpoints
    checkpoints = {}

    @staticmethod
    def point(name):
        # Record the current time with the given name in milliseconds
        Timer.checkpoints[name] = time.time() * 1000  # Convert to milliseconds

    @staticmethod
    def reset():


        numOfPoints = len(Timer.checkpoints)
        # Generate the report and clear checkpoints
        if not Timer.checkpoints:
            return "No checkpoints to report."

        report = []
        report.append("")
        report.append("Time report:")

        sorted_checkpoints = sorted(Timer.checkpoints.items(), key=lambda x: x[1])

        # Calculate the total time from the first to the last checkpoint
        total_time = sorted_checkpoints[-1][1] - sorted_checkpoints[0][1]
        report.append(f"Total time: {total_time:.2f} milliseconds")

        # Calculate the time between each checkpoint and their percentage of the total time
        for i in range(1, len(sorted_checkpoints)):
            name1, time1 = sorted_checkpoints[i - 1]
            name2, time2 = sorted_checkpoints[i]
            time_diff = time2 - time1
            percentage = (time_diff / total_time) * 100
            report.append(f"Time from {name1} to {name2}: {time_diff:.2f} milliseconds ({percentage:.2f}%)")

        # Clear the checkpoints for the next run
        Timer.checkpoints.clear()
        
        return "\n".join(report)