import argparse
import os

import numpy as np
import pandas as pd
import bilby
import matplotlib.pyplot as plt


from nmma.gw.gw_likelihood import GravitationalWaveTransientLikelihood
from bilby.gw.conversion import generate_posterior_samples_from_marginalized_likelihood, generate_all_bns_parameters


#####PARSER########

parser = argparse.ArgumentParser()

parser.add_argument("--source", help="integer, which source from events.dat to use", required=True)
parser.add_argument("--outdir", help="outdir for the result files", required=True)

parser.add_argument("--plot", help="whether to make plots, defaults to True", default=True)
parser.add_argument("--random-seed", help="bilby random seed", default=4576892, type=int)
parser.add_argument("--distance-marginalization", help="whether to perform distance marginalization. Defaults to True.", default=True)

###################

#####PARSING########

minimum_frequency = 5.
maximum_frequency = 2048.
reference_frequency = 5.
sampling_frequency = 4096

args = parser.parse_args()

bilby.core.utils.random.seed(args.random_seed)

###################

def main():
    
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    events = pd.read_csv("../events.dat", sep=" ")
    event = events.loc[int(args.source), ["mass_1", "mass_2", "chi_1", "chi_2", "lambda_1", "lambda_2", "theta_jn", "luminosity_distance", "phase", "psi", "ra", "dec", "geocent_time", "redshift"]]
    CHIEFF = (event['chi_1'] * event['mass_1'] + event['chi_2'] * event['mass_2']) / (event['mass_1'] + event['mass_2'])
    MCHIRP = bilby.gw.conversion.component_masses_to_chirp_mass(event['mass_1'], event["mass_2"])
    event["mass_ratio"] = event["mass_2"] / event["mass_1"]
    event["chirp_mass"] = MCHIRP
       
    duration = bilby.gw.utils.calculate_time_to_merger(frequency = minimum_frequency, mass_1 = event['mass_1'], mass_2 = event['mass_2'], chi=CHIEFF, safety=1.3)
    duration = int(duration) + 2.

    injection_parameters = event[["chirp_mass", "mass_ratio", "chi_1", "chi_2", "lambda_1", "lambda_2", "theta_jn", "luminosity_distance", "phase", "psi", "ra", "dec", "geocent_time"]].to_dict()

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXAS_NRTidalv3", 
                                reference_frequency=reference_frequency,
                                minimum_frequency=minimum_frequency),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
    )
    
    
    ###################
    # INTERFEROMETERS #
    ###################
    
    ifos = bilby.gw.detector.InterferometerList(["ETL1", "ETL2"])
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = maximum_frequency 
    
    ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
                                                       duration=duration,
                                                       start_time=injection_parameters["geocent_time"] - duration + 2)
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)


    #########################
    # PRIORS AND LIKELIHOOD #
    #########################   

    priors = bilby.core.prior.PriorDict(filename='./bns.prior')
    priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=MCHIRP-0.01, maximum=MCHIRP+0.01)
    
    # more efficient sampling for high mass ratios
    if injection_parameters["mass_ratio"]>=0.9:
        priors["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(name="mass_ratio", minimum=0.4, maximum=1., equal_mass=True)
    
    priors["geocent_time"] = bilby.gw.prior.Uniform(name='geocent_time', minimum=injection_parameters["geocent_time"]-0.1, maximum=injection_parameters["geocent_time"]+0.1)
    priors["ra"] = injection_parameters["ra"]
    priors["dec"] = injection_parameters["dec"]
    
    # make waveform generator for likelihood evaluations
    search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
        waveform_arguments=dict(waveform_approximant="IMRPhenomXAS_NRTidalv3",
                                reference_frequency=reference_frequency),
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters
    )
    
    # make multi-banded likelihood
    likelihood = GravitationalWaveTransientLikelihood(
        priors=priors,
        interferometers=ifos,
        waveform_generator=search_waveform_generator,
        gw_likelihood_type="MBGravitationalWaveTransient",
        reference_chirp_mass=priors["chirp_mass"].minimum,
        distance_marginalization=args.distance_marginalization,
        distance_marginalization_lookup_table = args.outdir + "/.distance_marginalization_lookup.npz",
        phase_marginalization=True,
        time_reference="geocent_time",
        reference_frame="sky",
        accuracy_factor=5,
    )

    ############
    # SAMPLING #
    ############
    
    
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=1024,
        naccept=60,
        npool=192,
        check_point_plot=False,
        check_point_delta_t=1800,
        print_method='interval-60',
        sample='acceptance-walk',
        injection_parameters=injection_parameters,
        outdir=args.outdir,
        label=args.outdir,
        save="hdf5"
    )

    posterior_samples = generate_all_bns_parameters(result.posterior)    
    posterior_samples = generate_posterior_samples_from_marginalized_likelihood(posterior_samples, likelihood.sub_model, npool=192, use_cache=False)
    posterior_samples = generate_all_bns_parameters(posterior_samples)
    np.savez(os.path.join(args.outdir, "posterior.npz"), **posterior_samples)

    true_redshift = event["redshift"]
    redshift_mean = np.random.normal(loc=true_redshift, scale=0.01*true_redshift, size=1)
    redshift_samples = np.random.normal(loc=redshift_mean, scale=0.01*true_redshift, size=posterior_samples["mass_1"].shape)

    posterior_samples["mass_1_source"] = posterior_samples["mass_1"] / (1 + redshift_samples)
    posterior_samples["mass_2_source"] = posterior_samples["mass_2"] / (1 + redshift_samples)
    np.savez(os.path.join(args.outdir, "posterior_mm.npz"), **posterior_samples)

    if args.plot:
        result.plot_corner()

        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        bins = np.linspace(0.9, 2.5, 100)

        posterior = np.load(os.path.join(args.outdir, "posterior.npz"))
        ax.hist(posterior["mass_1_source"], bins=bins, density=True, color="blue", histtype="step")
        ax.hist(posterior["mass_2_source"], bins=bins, density=True, color="orange", histtype="step")

        posterior_mm = np.load(os.path.join(args.outdir, "posterior_mm.npz"))
        ax.hist(posterior_mm["mass_1_source"], bins=bins, density=True, color="lightskyblue", histtype="step")
        ax.hist(posterior_mm["mass_2_source"], bins=bins, density=True, color="bisque", histtype="step")

        mass_1_source = event["mass_1"] / (1 + event["redshift"])
        mass_2_source = event["mass_2"] / (1 + event["redshift"])
        ax.vlines([mass_1_source, mass_2_source], *ax.get_ylim(), color="red")
        ax.set_xlabel("$m$ in source frame")

        fig.savefig(os.path.join(args.outdir, "mass_posterior.pdf"), dpi=200, bbox_inches="tight")
    
if __name__=="__main__":
    main()
