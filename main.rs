use std::collections::HashMap;
use rand::Rng; // Ensure you have this import for random number generation

// Define the ModelFreeLineWorld struct
pub struct ModelFreeLineWorld {
    nb_cells: usize,
    agent_pos: usize,
}

impl ModelFreeLineWorld {
    pub fn new(nb_cells: usize) -> Self {
        Self {
            nb_cells,
            agent_pos: nb_cells / 2,
        }
    }

    pub fn reset(&mut self) {
        self.agent_pos = self.nb_cells / 2;
    }

    pub fn reset_random_state(&mut self) {
        let mut rng = rand::thread_rng();
        self.agent_pos = rng.gen_range(0..self.nb_cells);
    }

    pub fn reset_random_state_2(&mut self) {
        let mut rng = rand::thread_rng();
        self.agent_pos = rng.gen_range(1..self.nb_cells - 1);
    }

    pub fn score(&self) -> f64 {
        if self.agent_pos == 0 {
            return -1.0;
        }
        if self.agent_pos == self.nb_cells - 1 {
            return 1.0;
        }
        0.0
    }

    pub fn state_id(&self) -> Option<()> {
        None // Placeholder for state ID logic
    }

    pub fn available_actions(&self) -> Vec<i32> {
        vec![0, 1]
    }

    pub fn step(&mut self, action: i32) {
        assert!(!self.is_game_over());
        assert!(self.available_actions().contains(&action));

        match action {
            0 => {
                self.agent_pos = self.agent_pos.saturating_sub(1); // Prevent underflow
            }
            1 => {
                self.agent_pos = (self.agent_pos + 1).min(self.nb_cells - 1); // Prevent overflow
            }
            _ => {}
        }
    }

    pub fn is_game_over(&self) -> bool {
        self.agent_pos == 0 || self.agent_pos == self.nb_cells - 1
    }
}

fn main() {
    // Adjustable variables
    let _epsilon = 0.1; // epsilon greedy-policy
    let _gamma = 0.9;   // discount factor
    let _alpha = 0.02;  // learning rate

    let _num_episodes = 50_000; // number of episodes to run
    let _eval_interval = 1_000;

    // Define the number of actions (like env.action_space.n in Python)
    let action_space_size = 4; // Example: say there are 4 possible actions

    // Create the Q table: a HashMap that maps a state (i32) to a Vec of zeros (for actions)
    let mut q_table: HashMap<i32, Vec<f64>> = HashMap::new();

    // Function to get or initialize the Q-values for a given state
    let mut get_q_values = |state: i32, action_space_size: usize| -> Vec<f64> {
        q_table
            .entry(state)
            .or_insert_with(|| vec![0.0; action_space_size])
            .clone() // return a clone of the vector of Q-values
    };

    // Define epsilon-greedy policy
    fn epsilon_greedy_policy(state: i32, epsilon: f64, action_space_size: usize) -> i32 {
        let mut rng = rand::thread_rng();
        if rng.gen_range(0.0..1.0) < epsilon {
            rng.gen_range(0..action_space_size)
        } else {
            state // Just returning the state here; modify as needed
        }
    }

    // Example usage
    let state = 1; // Example state
    let q_values = get_q_values(state, action_space_size);

    println!("{:?}", q_values); // Outputs: [0.0, 0.0, 0.0, 0.0]
}
