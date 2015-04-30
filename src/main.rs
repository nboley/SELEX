// Argument parsing
use std::env;

use std::error::Error;

// File parsing
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;

// Random number generation
extern crate rand;
use rand::distributions::{IndependentSample, Range};

// kJ/mol*K
static R: f32 = 8.314472e-3;
// K
static T: f32 = 300.0;
static NUM_SIMS_PER_RND: i32 = 10000;

#[derive(Debug)]
struct Score {
    a: f32,
    c: f32,
    g: f32,
    t: f32
}

#[derive(Debug)]
enum Base{ A, C, G, T, N }

#[derive(Debug)]
struct SelexRead {
    seq: Vec<Base>,
    round: i32,
    cnt: i32,
}

#[derive(Debug)]
struct BindingModel {
    consensus_gfe: f32,
    pos_gfes: Vec<Score>
}


/*
fn update_weights(rd: &SelexRead, 
                  base_scores: &Vec<Score>,
                  unbnd_conc: f32) -> Vec<f32> {
    let bs_iter = rd.seq.windows(base_scores.len());
    let mut weights = Vec::with_capacity(bs_iter.len());

    let mut total_occupancy = 0.0;

    for substr in bs_iter {
        let occupancy = calc_occupancy(substr, base_scores, unbnd_conc);
        total_occupancy += occupancy;
        weights.push( occupancy );
    }
    
    for weight in weights.iter_mut() {
        *weight /= total_occupancy;
    }
    
    weights
}

#[derive(Debug)]
struct SelexReadAndWeights {
    rd: SelexRead,
    weights: Vec<f32>,
}
*/

fn load_sequences_in_round(fname: &str, round: i32) -> Vec<SelexRead> {
    let file = match File::open(&fname) {
        // The `desc` field of `IoError` is a string that describes the error
        Err(why) => panic!(
            "couldn't open {}: {}", fname, Error::description(&why) ),
        Ok(file) => file,
    };

    let mut seqs: Vec<SelexRead> = Vec::new();
    
    for line in BufReader::new(&file).lines() {
        let res = line.unwrap();
        let mut line_iter = res.split("\t");

        let mut seq = Vec::new();
        for char in line_iter.next().unwrap().chars() {
            let base = match char {
                'A' => Base::A,
                'C' => Base::C,
                'G' => Base::G,
                'T' => Base::T,
                _ => Base::N,
            };
            seq.push(base); 
        }

        let cnt: i32  = line_iter.next().unwrap_or("1").parse().unwrap_or(1);
        let rd = SelexRead{seq: seq, round: round, cnt: cnt};
        seqs.push(rd);
    };

    seqs
}


fn calc_fwd_and_revcmp_gibbs_free_energy(
        seq: &[Base], bm: &BindingModel) -> (f32, f32) {
    assert!(seq.len() == bm.pos_gfes.len());
    let mut score: f32 = bm.consensus_gfe;
    let mut rc_score: f32 = bm.consensus_gfe;

    for pos in 0..bm.pos_gfes.len() {
        score += match seq[pos] {
            Base::A => bm.pos_gfes[pos].a,
            Base::C => bm.pos_gfes[pos].c,
            Base::G => bm.pos_gfes[pos].g,
            Base::T => bm.pos_gfes[pos].t,
            Base::N => unreachable!(),
        };

        rc_score += match seq[bm.pos_gfes.len() - pos - 1] {
            Base::A => bm.pos_gfes[pos].t,
            Base::C => bm.pos_gfes[pos].g,
            Base::G => bm.pos_gfes[pos].c,
            Base::T => bm.pos_gfes[pos].a,
            Base::N => unreachable!(),
        };

    }    
    (score, rc_score)
}

fn calc_occupancy(gibbs_free_energy: f32, 
                  unbnd_conc: f32) -> f32 { 
    let numerator = unbnd_conc*(gibbs_free_energy/(R*T)).exp();
    numerator/(1.0+numerator)
}

fn calc_max_occupancy(seq: &[Base], 
                      bm: &BindingModel,
                      unbnd_conc: f32) -> f32 {
    let mut max_occupancy = 0.0;
    
    for sub_seq in seq.windows(bm.pos_gfes.len()) {
        let (gfe, rc_gfe) = calc_fwd_and_revcmp_gibbs_free_energy(
            sub_seq, bm);
        let fwd_occupancy = calc_occupancy(gfe, unbnd_conc);
        let rc_occupancy = calc_occupancy(rc_gfe, unbnd_conc);

        //print!("{:?}\t{}:{}\t{}:{}\n", sub_seq, gfe, fwd_occupancy, rc_gfe, rc_occupancy);

        if fwd_occupancy > max_occupancy {
            max_occupancy = fwd_occupancy;
        };

        if rc_occupancy > max_occupancy {
            max_occupancy = rc_occupancy;
        };
    }
    //print!("\n");
    max_occupancy
}

fn simulate_read(len: usize) -> Vec<Base> {
    let mut rng = rand::thread_rng();
    let range_iter = Range::new(0, 4);
    let mut seq = Vec::with_capacity(len);
    for _ in 0..len {
        let base = match range_iter.ind_sample(&mut rng) {
            0 => Base::A,
            1 => Base::C,
            2 => Base::G,
            3 => Base::T,
            _ => unreachable!()
        };
        seq.push(base);
    }
    seq
}

fn load_all_reads() -> Vec<SelexRead> {
    let mut observed_reads: Vec<SelexRead> = Vec::new();
        
    for fname in env::args().skip(1) {
        let round: i32 = 
            fname.split("_").last().unwrap(
                ).split(".").next().unwrap(
                ).parse().unwrap();
        
        for rd in load_sequences_in_round(&fname, round) {
            observed_reads.push( rd );
        }

        print!("Loaded {} (Round {})\n", fname, round);
    };
    
    observed_reads
}

/*
fn load_uniform_score_matrix(motif_len: i32) -> Vec<Score> {
    let mut score_matrix = Vec::with_capacity(motif_len as usize);
    for _ in 0..motif_len {
        score_matrix.push(Score{a: -1.0, c: -1.0, g: -1.0, t: -1.0})
    }
    score_matrix
}
*/

fn load_initial_binding_model() -> BindingModel {
    let mut bm = BindingModel{ 
        consensus_gfe: 40.0, pos_gfes: Vec::new() };
    bm.pos_gfes.push(Score{a: -10.0, c: -10.0, g: -10.0, t: 0.0});
    bm.pos_gfes.push(Score{a: 0.0, c: -10.0, g: -10.0, t: -10.0});
    bm.pos_gfes.push(Score{a: 0.0, c: -10.0, g: -10.0, t: -10.0});
    bm.pos_gfes.push(Score{a: -10.0, c: -10.0, g: -10.0, t: 0.00});
    bm.pos_gfes.push(Score{a: -2.0, c: 0.0, g: -2.0, t: -2.0});
    bm.pos_gfes.push(Score{a: -2.0, c: 0.0, g: -2.0, t: -2.0});
    bm.pos_gfes.push(Score{a: -2.0, c: 0.0, g: -2.0, t: -2.0});
    bm.pos_gfes.push(Score{a: -2.0, c: 0.0, g: -2.0, t: -2.0});

    bm
}

fn estimate_average_occupancies(
    obs_rds: &Vec<SelexRead>, 
    bm: &BindingModel, 
    unbnd_concs: &Vec<f32>
        ) -> Vec<f32> {
    // Find the number of reads in each round
    let mut rnd_cnts = vec![0; unbnd_concs.len()];
    for rd in obs_rds.iter() {
        rnd_cnts[rd.round as usize] += 1;
    };
    
    // Initialize a vector to hold the normalization factors
    let mut total_occupancies = vec![0.0; unbnd_concs.len()];

    // Loop through the observed round 1 reads to make sure that we get the seq 
    // length distribution correct
    let mut num_sim_reads = 0;
    while num_sim_reads < NUM_SIMS_PER_RND {
        for rd in obs_rds.iter() {
            num_sim_reads += 1;
            
            let sim_seq = simulate_read(rd.seq.len());
            print!("{:?}\t", sim_seq );

            let mut curr_occupancy = 1.0;
            for (rnd, unbnd_conc) in unbnd_concs.iter().enumerate() {
                let max_occ = calc_max_occupancy(
                    &(sim_seq[..]), &bm, *unbnd_conc);
                
                curr_occupancy *= max_occ;
                total_occupancies[rnd] += curr_occupancy;
                print!("{}:{}\t", rnd, max_occ );
            };
            print!("\n" );
            if num_sim_reads >= NUM_SIMS_PER_RND { break; }
        };
    };

    let mut mean_occupancies = Vec::new();
    
    for (i, val) in total_occupancies.iter().enumerate() {
        let prev_conc = if i == 0 {NUM_SIMS_PER_RND as f32} else { total_occupancies[i-1] };
        print!("{}\t{}\t{}\t{}\n", i, prev_conc, val, val/prev_conc );
        mean_occupancies.push( val/prev_conc );
    }
    mean_occupancies
}

fn calc_consensus_seq(bm: &BindingModel) -> Vec<Base> {
// Calculate the concensus sequence and it's maximum occupancy
    let mut consensus_seq = Vec::new();
    
    for score in bm.pos_gfes.iter() {
        let mut curr_base = Base::A;
        let mut curr_val = score.a;
        if  score.c > curr_val {
            curr_val = score.c;
            curr_base = Base::C;
        }
        if score.g > curr_val {
            curr_val = score.g;
            curr_base = Base::G;
        }
        if score.t > curr_val {
            // Dont need this
            //curr_val = score.t;
            curr_base = Base::T;
        }
        consensus_seq.push(curr_base)
    }
    
    consensus_seq
}

fn calc_unbnd_conc_for_specified_max_occupancy(
    bm: &BindingModel, consensus_occ: f32, mean_occ: f32) -> f32 {
// Given a binding model, find what the concensus energy and 
// unbound concentration would need to be to produce the 
// specified occupancies
    let concensus_seq = calc_consensus_seq(bm);
    let (fwd_gfe, rc_gfe) = calc_fwd_and_revcmp_gibbs_free_energy(
        &concensus_seq, bm );
    let mut gfe = fwd_gfe;
    if rc_gfe > gfe { gfe = rc_gfe };

    let norm_gfe = gfe/(R*T);
    let unbnd_conc = (consensus_occ/(1.0-consensus_occ))*norm_gfe.exp();

    let occupancy = calc_max_occupancy(&concensus_seq, bm, unbnd_conc);

    print!("GFE: {}\tUnbnd Conc: {}\tCalc Occ:{}\n", 
           gfe, unbnd_conc, occupancy);
    unbnd_conc
}


fn calc_lhd(
    obs_rds: &Vec<SelexRead>, 
    bm: &BindingModel, 
    unbnd_concs: &Vec<f32>
        ) -> f32
{
    let mut lhd_num = 0.0; 
    for rd in obs_rds.iter() {
        let unbnd_conc = unbnd_concs[rd.round as usize];
        let log_occ: f32 = calc_max_occupancy(
            &(rd.seq[..]), bm, unbnd_conc).ln();
        lhd_num += (rd.round as f32)*log_occ;
    }

    return lhd_num;
    
    /*
    let norm_constant = estimate_normalization_constants(
        &obs_rds, &score_mat, &unbnd_concs);

    let mut norm_constant = 0.0;
    for (i, constant) in norm_constants.iter_mut().enumerate() {
        norm_constant += constant*(rnd_cnts[i] as f32)/(
            NUM_SIMS_PER_RND as f32 );
    };

    lhd_num - norm_constant
    */
}

/*
TODO:
0) get better dg mnatrix
1) implement simulator
2) use reverse complement

*/
 
fn main() {
    // 100-250 ng of soluble protein
    // 100 ng of selection ligand
    // 83 ng of non-sepcific ligand
    // all in 20 ul of h20
    
    // ug/L
    let tot_prot_conc = 200.0;
    let tot_DNA_conc = 150.0;
 
    let bm = load_initial_binding_model();

    let observed_reads = load_all_reads();

    let unbnd_conc = calc_unbnd_conc_for_specified_max_occupancy(
        &bm, 0.99);
    let sim_rd = simulate_read(bm.pos_gfes.len());
    let random_occ = calc_max_occupancy(&sim_rd, &bm, unbnd_conc);
    //print!( "{}\t{}\n", 0.99, random_occ);
    //return;
    
    let unbnd_concs = vec![unbnd_conc; env::args().len()];

    let mean_occs = estimate_average_occupancies(
        &observed_reads, &bm, &unbnd_concs);

    let concensus_seq = calc_consensus_seq(&bm);

    print!("{:?}\t{:?}\n", concensus_seq, mean_occs);

    for _ in 1..2 {
        let lhd = calc_lhd(
            &observed_reads, &bm, &unbnd_concs);
        print!("Lhd: {}\n", lhd);
    }
}
