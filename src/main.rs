use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet};

struct ModelFreeLineWorld {
    nb_cells: usize,
    agent_pos: usize,
}

impl ModelFreeLineWorld {
    // Constructor to initialize the environment with a given number of cells
    fn new(nb_cells: usize) -> Self {
        Self {
            nb_cells,
            agent_pos: nb_cells / 2, // Start in the middle cell
        }
    }

    // Reset the agent's position to the middle of the cells
    fn reset(&mut self) {
        self.agent_pos = self.nb_cells / 2;
    }

    // Reset the agent's position to a random state (full range)
    fn reset_random_state(&mut self) {
        let mut rng = rand::thread_rng();
        self.agent_pos = rng.gen_range(0..self.nb_cells);
    }

    // Reset the agent's position to a random state (excluding the terminal states)
    fn reset_random_state_excluding_terminal(&mut self) {
        let mut rng = rand::thread_rng();
        self.agent_pos = rng.gen_range(1..self.nb_cells - 1);
    }

    // Check if the game is over (agent is at a terminal state)
    fn is_game_over(&self) -> bool {
        self.agent_pos == 0 || self.agent_pos == self.nb_cells - 1
    }

    // Return the score based on the agent's current position
    fn score(&self) -> f64 {
        if self.agent_pos == 0 {
            -1.0 // Lose
        } else if self.agent_pos == self.nb_cells - 1 {
            1.0 // Win
        } else {
            0.0 // Neutral
        }
    }

    // Return the available actions (0 for left, 1 for right)
    fn available_actions(&self) -> Vec<usize> {
        vec![0, 1]
    }

    // Take a step based on the chosen action
    fn step(&mut self, action: usize) {
        assert!(!self.is_game_over()); // Ensure the game is not over
        assert!(self.available_actions().contains(&action)); // Ensure the action is valid

        // Update the agent's position based on the action
        match action {
            0 => self.agent_pos -= 1, // Move left
            1 => self.agent_pos += 1, // Move right
            _ => panic!("Invalid action"), // Handle invalid actions
        }
    }

    // Display the current state of the environment
    fn display(&self) {
        println!("Is Game Over: {}", self.is_game_over());
        let mut display_lst = String::new();
        for pos in 0..self.nb_cells {
            if pos == self.agent_pos {
                display_lst.push('X'); // Represent the agent's position
            } else {
                display_lst.push('_'); // Represent empty cells
            }
        }
        println!("{}", display_lst); // Print the visual representation
    }
}

// Function to choose a random policy (randomly select an action)
fn random_policy() -> usize {
    let actions = vec![0, 1];
    let mut rng = rand::thread_rng();
    *actions.choose(&mut rng).unwrap() // Return a random action
}

// Monte Carlo Agent struct
struct MonteCarloAgent {
    env: ModelFreeLineWorld,
    episodes: usize,
    gamma: f64,
    epsilon: f64,
    Q: HashMap<(usize, usize), f64>,
    returns: HashMap<(usize, usize), Vec<f64>>,
}

impl MonteCarloAgent {
    // Constructor for the Monte Carlo Agent
    fn new(env: ModelFreeLineWorld, episodes: usize, gamma: f64, epsilon: f64) -> Self {
        let mut Q = HashMap::new();
        let mut returns = HashMap::new();
        for state in 0..env.nb_cells {
            for action in env.available_actions() {
                Q.insert((state, action), 0.0);
                returns.insert((state, action), Vec::new());
            }
        }

        Self {
            env,
            episodes,
            gamma,
            epsilon,
            Q,
            returns,
        }
    }

    // Epsilon-greedy policy implementation
    fn epsilon_greedy_policy(&self, state: usize) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            *self.env.available_actions().choose(&mut rng).unwrap() // Random action
        } else {
            // Action with max Q-value
            *self
                .env
                .available_actions()
                .iter()
                .max_by(|&&a, &&b| {
                    self.Q[&(state, a)]
                        .partial_cmp(&self.Q[&(state, b)])
                        .unwrap()
                })
                .unwrap()
        }
    }

    // Generate an episode
    fn generate_episode(&mut self) -> Vec<(usize, usize, f64)> {
        let mut episode = Vec::new();
        self.env.reset_random_state();

        while !self.env.is_game_over() {
            let state = self.env.agent_pos;
            let action = self.epsilon_greedy_policy(state);
            self.env.step(action);
            let reward = self.env.score();
            episode.push((state, action, reward));
        }

        episode
    }

    // Update Q-values based on the episode
    fn update_Q_values(&mut self, episode: Vec<(usize, usize, f64)>) {
        let mut G = 0.0; // Total return
        let mut visited_state_action_pairs = HashSet::new();

        for (state, action, reward) in episode.iter().rev() {
            G = self.gamma * G + reward; // Discounted return
            if !visited_state_action_pairs.contains(&(*state, *action)) {
                self.returns.get_mut(&(*state, *action)).unwrap().push(G);
                // Update Q-value as the average of returns
                let avg = self.returns[&(*state, *action)]
                    .iter()
                    .copied()
                    .sum::<f64>()
                    / self.returns[&(*state, *action)].len() as f64;
                self.Q.insert((*state, *action), avg);
                visited_state_action_pairs.insert((*state, *action));
            }
        }
    }

    // Train the agent
    fn train(&mut self) {
        for _ in 0..self.episodes {
            let episode = self.generate_episode();
            self.update_Q_values(episode);
        }
    }
}

fn main() {
    let nb_cells = 10;
    let env = ModelFreeLineWorld::new(nb_cells);
    let mut agent = MonteCarloAgent::new(env, 1000, 0.9, 0.1);
    agent.train();

    // Print final Q-values
    println!("Final Q-values:");
    for state in 0..agent.env.nb_cells {
        for action in agent.env.available_actions() {
            println!(
                "Q(cell: {}, step: {}) = {}",
                state,
                if action == 0 { "left" } else { "right" },
                agent.Q[&(state, action)]
            );
        }
    }
}
