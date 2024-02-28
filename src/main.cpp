#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>

#include "exec/GPU_Parameter_Set.h"
#include "macsim/macsim.h"
#include "utils/rapidxml/rapidxml.hpp"
#include "ram.h"

using namespace std;

/**
 * Parse command line arguments
 */
void command_line_args(char* argv[], string& gpuconfig_file_path, string& kernel_config_file_path)
{
	for (int arg_cntr = 1; arg_cntr < 9; arg_cntr++) {
		if (argv[arg_cntr] == nullptr) break;
		string arg = argv[arg_cntr];

		char gpuconfig_path_switch[] = "-g";
		if (arg.compare(0, strlen(gpuconfig_path_switch), gpuconfig_path_switch) == 0) {
			gpuconfig_file_path.assign(argv[++arg_cntr]);
			continue;
		}

		// -t is optional. if it is not specified, it will use the kernel_config path in gpuconfig.xml
		char kernel_config_path_switch[] = "-t";
		if (arg.compare(0, strlen(kernel_config_path_switch), kernel_config_path_switch) == 0) {
			kernel_config_file_path.assign(argv[++arg_cntr]);
			continue;
		}
	}
}

/**
 * Reads GPU configuration from specified XML file
*/
void read_gpu_configuration_parameters(const string gpu_config_file_path, GPU_Parameter_Set* gpu_params)
{
	ifstream gpu_config_file;
	gpu_config_file.open(gpu_config_file_path.c_str());

	if (!gpu_config_file) {
		PRINT_MESSAGE("The specified GPU configuration file does not exist.")
		PRINT_MESSAGE("Using Macsim's default configuration.")
		PRINT_MESSAGE("Writing the default configuration parameters to the expected configuration file.")

		Utils::XmlWriter xmlwriter;
		string tmp;
		xmlwriter.Open(gpu_config_file_path.c_str());

		printf("Default GPU Configuration File is %s\n",gpu_config_file_path.c_str());
		gpu_params->XML_serialize(xmlwriter);
		xmlwriter.Close();
		PRINT_MESSAGE("[====================] Done!\n")
	} else {
		//Read input workload parameters
		string line((std::istreambuf_iterator<char>(gpu_config_file)),
			std::istreambuf_iterator<char>());
		gpu_config_file >> line;
		if (line.compare("USE_INTERNAL_PARAMS") != 0) {
			rapidxml::xml_document<> doc;    // character type defaults to char
			char temp_string [line.length() + 1];
			strcpy(temp_string, line.c_str());
			doc.parse<0>(temp_string);
			rapidxml::xml_node<> *macsim_config = doc.first_node("GPU_Parameter_Set");
			if (macsim_config != NULL) {
				gpu_params = new GPU_Parameter_Set;
				gpu_params->XML_deserialize(macsim_config);
			} else {
				PRINT_MESSAGE("Error in the GPU configuration file!")
				PRINT_MESSAGE("Using Macsim's default configuration.")
			}
		} else {
			PRINT_MESSAGE("Using Macsim's default configuration.")
			PRINT_MESSAGE("Writing the default configuration parameters to the expected configuration file.")

			Utils::XmlWriter xmlwriter;
			string tmp;
			xmlwriter.Open(gpu_config_file_path.c_str());
			gpu_params->XML_serialize(xmlwriter);
			xmlwriter.Close();
			PRINT_MESSAGE("[====================] Done!\n")
		}
	}

	gpu_config_file.close();
}

/**
 * Prints usage help message
*/
void print_help()
{
	cout << 
	"MacSim\n"
	"\tUsage: macsim [-g gpu-config-file] [-t kernel-config-file]\n" << endl;
}

/**
 * Main
*/
int main(int argc, char* argv[])
{
	// Parse command line argumnets
	string ssd_config_file_path, gpu_config_file_path, output_stats_file_path, kernel_config_file_path;
	if (argc != 5 && argc != 1) {
		print_help();
		return 1;
	}
	command_line_args(argv, gpu_config_file_path, kernel_config_file_path);

	// Read config files
	GPU_Parameter_Set* gpu_params = new GPU_Parameter_Set;
	printf("Reading GPU Configurations...");
	read_gpu_configuration_parameters(gpu_config_file_path, gpu_params);
	
	if (!kernel_config_file_path.empty()) {
		gpu_params->GPU_Trace_Path = kernel_config_file_path;
	}

	// Setup memory request and response queues
	queue<RAM_request> *ram_request_queue = new queue<RAM_request>();
	queue<RAM_response> *ram_response_queue = new queue<RAM_response>();
	
	// Setup sim components
	RAM ram(200);
	macsim gpu(gpu_params);

	// Attach queues
	ram.set_queues(ram_request_queue, ram_response_queue);
	gpu.set_queues(ram_request_queue, ram_response_queue);
	
	// Start simulation
	time_t start_time = time(0);
	char* dt = ctime(&start_time);
	PRINT_MESSAGE("Macsim started @ " << dt);
	PRINT_MESSAGE("**************************************************");

	while(gpu.run_a_cycle()) {
		if (gpu.m_cycle % 100000 == 0) {
			printf("[Cycle: %lu]: mem_requests: %d, mem_responses: %d, avg_latency: %u\n", gpu.m_cycle, gpu.get_n_requests(), gpu.get_n_responses(), gpu.get_avg_latency());
		}
		ram.run_a_cycle();
	}

	// End simulation
	time_t end_time = time(0);
	dt = ctime(&end_time);
	PRINT_MESSAGE("**************************************************");
	PRINT_MESSAGE("Simulation finished @ " << dt)
	uint64_t duration = (uint64_t)difftime(end_time, start_time);
	PRINT_MESSAGE("Total simulation time: " << duration / 3600 << ":" << (duration % 3600) / 60 << ":" << ((duration % 3600) % 60))
	
	gpu.print_stats();
	gpu.end_sim();

	delete gpu_params;
	delete ram_request_queue;
	delete ram_response_queue;
	return 0;
}