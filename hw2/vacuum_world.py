import numpy as np


class VacuumEnvironment:
    def __init__(self, room_a_status, room_b_status, agent_initial_location):
        self.room_a_status = room_a_status
        self.room_b_status = room_b_status
        self.agent_location = agent_initial_location

    def update_environment(self, action: str):
        # The environment only needs to be updated if the agent cleans a room
        if self.agent_location == "a" and action == "suck":
            self.room_a_status = "clean"
        if self.agent_location == "b" and action == "suck":
            self.room_b_status = "clean"

    def update_agent_location(self, action: str):
        # The agent's location is only updated if it moves right from Room A or left from Room B, otherwise it runs into
        # a wall and stays where it is
        if self.agent_location == "a" and action == "right":
            self.agent_location = "b"
        if self.agent_location == "b" and action == "left":
            self.agent_location = "a"

    def get_room_status(self, room: str):
        if room == "a":
            return self.room_a_status
        if room == "b":
            return self.room_b_status

    def get_agent_location(self):
        return self.agent_location

    def print_environment_status(self):
        print(f"[ENV]: room a status: {self.room_a_status}, room b status: {self.room_b_status}, agent location: room {self.agent_location}")


class ReflexAgent:
    def __init__(self, agent_initial_location: str):
        self.location = agent_initial_location
        self.location_status = None

    def perceive(self, environment: VacuumEnvironment):
        self.location = environment.get_agent_location()
        self.location_status = environment.get_room_status(self.location)
        print(f"[AGENT]: I'm in room {self.location} and the room is {self.location_status}")

    def act(self):
        # If current location (room) is dirty, clean it
        if self.location_status == "dirty":
            print(f"[AGENT]: Chosen action: suck")
            return "suck"
        # Otherwise, move to the other room Location can either be in Room A or Room B
        else:
            if self.location == "a":
                print(f"[AGENT]: Chosen action: right")
                return "right"
            if self.location == "b":
                print(f"[AGENT]: Chosen action: left")
                return "left"


def performance_measure(environment: VacuumEnvironment):
    performance = 0
    if environment.room_a_status == "clean":
        performance += 1
    if environment.room_b_status == "clean":
        performance += 1
    return performance


def main_loop(room_a_status, room_b_status, agent_initial_location):
    environment = VacuumEnvironment(room_a_status, room_b_status, agent_initial_location)
    environment.print_environment_status()
    
    agent = ReflexAgent(agent_initial_location)

    total_performance = 0

    time = 0
    while time < 1000:
        # Agent performance is recorded at beginning of timestep (see brief)
        performance = performance_measure(environment)
        total_performance += performance
        print(f"TIMESTEP {time + 1}: Performance = {performance}, Total Performance = {total_performance}")
        
        # Agent perceives and acts in the environment
        agent.perceive(environment)
        action = agent.act()

        # The environment and the agent's location within it are updated
        environment.update_environment(action)
        environment.update_agent_location(action)
        environment.print_environment_status()

        time += 1
    print("DONE!")

    return total_performance


if __name__ == "__main__":
    total_performance_array = []
    
    # Configuration 1: Room A dirty, Room B dirty, Agent in Room A
    room_a_status = "dirty"
    room_b_status = "dirty"
    agent_initial_location = "a"
    total_performance = main_loop(room_a_status, room_b_status, agent_initial_location)
    total_performance_array.append(total_performance)

    # Configuration 3: Room A dirty, Room B clean, Agent in Room A
    room_a_status = "dirty"
    room_b_status = "clean"
    agent_initial_location = "a"
    total_performance = main_loop(room_a_status, room_b_status, agent_initial_location)
    total_performance_array.append(total_performance)
    
    # Configuration 5: Room A clean, Room B dirty, Agent in Room A
    room_a_status = "clean"
    room_b_status = "dirty"
    agent_initial_location = "a"
    total_performance = main_loop(room_a_status, room_b_status, agent_initial_location)
    total_performance_array.append(total_performance)
    
    # Configuration 1: Room A clean, Room B clean, Agent in Room A
    room_a_status = "clean"
    room_b_status = "clean"
    agent_initial_location = "a"
    total_performance = main_loop(room_a_status, room_b_status, agent_initial_location)
    total_performance_array.append(total_performance)

    print(f"Total Performance array:\n {total_performance_array}")
    print(f"Average Total Performance: {np.mean(total_performance_array)}")

