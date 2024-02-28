#include "GPU_Parameter_Set.h"
#include <algorithm>

int GPU_Parameter_Set::Cycle_Per_Period = 10000;
int GPU_Parameter_Set::Num_Of_Cores = 4;
int GPU_Parameter_Set::Max_Block_Per_Core = 4;
Block_Scheduling_Policy_Types GPU_Parameter_Set::Block_Scheduling_Policy = Block_Scheduling_Policy_Types::ROUND_ROBIN;
Warp_Scheduling_Policy_Types GPU_Parameter_Set::Warp_Scheduling_Policy = Warp_Scheduling_Policy_Types::ROUND_ROBIN;
int GPU_Parameter_Set::Block_Stride = 1;
double GPU_Parameter_Set::Compiler_Stride = 4;
std::string GPU_Parameter_Set::GPU_Trace_Path = "/fast_data/echung67/trace/nvbit/backprop/128/kernel_config.txt";
int GPU_Parameter_Set::N_Repeat = 1;
bool GPU_Parameter_Set::Enable_GPU_Cache = true;
bool GPU_Parameter_Set::GPU_Cache_Log = false;
bool GPU_Parameter_Set::Enable_GPU_Vaddr_Dump = false;
int GPU_Parameter_Set::L1Cache_Size = 8;
int GPU_Parameter_Set::L1Cache_Assoc = 2;
int GPU_Parameter_Set::L1Cache_Line_Size = 64;
int GPU_Parameter_Set::L1Cache_Banks = 1;
int GPU_Parameter_Set::L2Cache_Size = 128;
int GPU_Parameter_Set::L2Cache_Assoc = 8;
int GPU_Parameter_Set::L2Cache_Line_Size = 64;
int GPU_Parameter_Set::L2Cache_Banks = 1;
int GPU_Parameter_Set::Sharing_Size = -1;
int GPU_Parameter_Set::Sharing_Stride = -1;

void GPU_Parameter_Set::XML_serialize(Utils::XmlWriter& xmlwriter)
{
	std::string tmp;
	tmp = "GPU_Parameter_Set";
	xmlwriter.Write_open_tag(tmp);

	std::string attr = "Cycle_Per_Period";
	std::string val = std::to_string(Cycle_Per_Period);
	xmlwriter.Write_attribute_string(attr, val);

	attr = "Num_Of_Cores";
	val = std::to_string(Num_Of_Cores);
	xmlwriter.Write_attribute_string(attr, val);

	attr = "Max_Block_Per_Core";
	val = std::to_string(Max_Block_Per_Core);
	xmlwriter.Write_attribute_string(attr, val);

	attr = "Scheduling_Policy";
	// val;
	switch (Block_Scheduling_Policy) {
		case Block_Scheduling_Policy_Types::ROUND_ROBIN:
			val = "ROUND_ROBIN";
			break;
		// case Block_Scheduling_Policy_Types::LARGE_CHUNK:
		// 	val = "LARGE_CHUNK";
			break;
		default:
			break;
	}
	xmlwriter.Write_attribute_string(attr, val);

	attr = "Block_Stride";
	val = std::to_string(Block_Stride);
	xmlwriter.Write_attribute_string(attr, val);

	attr = "Compiler_Stride";
	val = std::to_string(Compiler_Stride);
	xmlwriter.Write_attribute_string(attr, val);

	attr = "GPU_Trace_Path";
	val = GPU_Trace_Path;
	xmlwriter.Write_attribute_string(attr, val);

	attr = "N_Repeat";
	val = std::to_string(N_Repeat);
	xmlwriter.Write_attribute_string(attr, val);

	attr = "Enable_GPU_Cache";
	val = (Enable_GPU_Cache ? "true" : "false");
	xmlwriter.Write_attribute_string(attr, val);

	attr = "GPU_Cache_Log";
	val = (GPU_Cache_Log ? "true" : "false");
	xmlwriter.Write_attribute_string(attr, val);

	attr = "Enable_GPU_Vaddr_Dump";
	val = (Enable_GPU_Vaddr_Dump ? "true" : "false");
	xmlwriter.Write_attribute_string(attr, val);

	attr = "L1Cache_Size";
	val = std::to_string(L1Cache_Size);
	xmlwriter.Write_attribute_string(attr, val);

	attr = "L1Cache_Assoc";
	val = std::to_string(L1Cache_Assoc);
	xmlwriter.Write_attribute_string(attr, val);

  attr = "L1Cache_Line_Size";
  val = std::to_string(L1Cache_Line_Size);
	xmlwriter.Write_attribute_string(attr, val);

  attr = "L1Cache_Banks";
  val = std::to_string(L1Cache_Banks);
	xmlwriter.Write_attribute_string(attr, val);

  attr = "L2Cache_Size";
  val = std::to_string(L2Cache_Size);
	xmlwriter.Write_attribute_string(attr, val);

  attr = "L2Cache_Assoc";
  val = std::to_string(L2Cache_Assoc);
	xmlwriter.Write_attribute_string(attr, val);

  attr = "L2Cache_Line_Size";
  val = std::to_string(L2Cache_Line_Size);
	xmlwriter.Write_attribute_string(attr, val);

  attr = "L2Cache_Banks";
  val = std::to_string(L2Cache_Banks);
	xmlwriter.Write_attribute_string(attr, val);

	xmlwriter.Write_close_tag();
}

void GPU_Parameter_Set::XML_deserialize(rapidxml::xml_node<> *node)
{
	try
	{
		for (auto param = node->first_node(); param; param = param->next_sibling()) {
			if (strcmp(param->name(), "Cycle_Per_Period") == 0) {
				std::string val = param->value();
				Cycle_Per_Period = std::stoi(val);
			} else if (strcmp(param->name(), "Num_Of_Cores") == 0) {
				std::string val = param->value();
				Num_Of_Cores = std::stoi(val);
			} else if (strcmp(param->name(), "Max_Block_Per_Core") == 0) {
				std::string val = param->value();
				Max_Block_Per_Core = std::stoi(val);
			} else if (strcmp(param->name(), "Block_Scheduling_Policy") == 0) {
				std::string val = param->value();
				std::transform(val.begin(), val.end(), val.begin(), ::toupper);
				if (strcmp(val.c_str(), "ROUND_ROBIN") == 0) {
					Block_Scheduling_Policy = Block_Scheduling_Policy_Types::ROUND_ROBIN;
				} else {
					PRINT_ERROR("Unknown block scheduling policy type specified in the GPU configuration file")
				}
			} else if (strcmp(param->name(), "Warp_Scheduling_Policy") == 0) {
				std::string val = param->value();
				std::transform(val.begin(), val.end(), val.begin(), ::toupper);
				if (strcmp(val.c_str(), "ROUND_ROBIN") == 0) {
					Warp_Scheduling_Policy = Warp_Scheduling_Policy_Types::ROUND_ROBIN;
				} else if (strcmp(val.c_str(), "GTO") == 0) {
					Warp_Scheduling_Policy = Warp_Scheduling_Policy_Types::GTO;
				} else if (strcmp(val.c_str(), "CCWS") == 0) {
					Warp_Scheduling_Policy = Warp_Scheduling_Policy_Types::CCWS;
				} else {
					PRINT_ERROR("Unknown warp scheduling policy type specified in the GPU configuration file")
				}
			} else if (strcmp(param->name(), "Block_Stride") == 0) {
				std::string val = param->value();
				Block_Stride = std::stoi(val);
			} else if (strcmp(param->name(), "Compiler_Stride") == 0) {
				std::string val = param->value();
				Compiler_Stride = std::stod(val);
			} else if (strcmp(param->name(), "GPU_Trace_Path") == 0) {
				std::string val = param->value();
				GPU_Trace_Path = val;
			} else if (strcmp(param->name(), "N_Repeat") == 0) {
				std::string val = param->value();
				N_Repeat = std::stoi(val);
			} else if (strcmp(param->name(), "Enable_GPU_Cache") == 0) {
				std::string val = param->value();
				std::transform(val.begin(), val.end(), val.begin(), ::toupper);
				Enable_GPU_Cache = (val.compare("FALSE") == 0 ? false : true);
			} else if (strcmp(param->name(), "GPU_Cache_Log") == 0) {
				std::string val = param->value();
				std::transform(val.begin(), val.end(), val.begin(), ::toupper);
				GPU_Cache_Log = (val.compare("FALSE") == 0 ? false : true);
			} else if (strcmp(param->name(), "Enable_GPU_Vaddr_Dump") == 0) {
				std::string val = param->value();
				std::transform(val.begin(), val.end(), val.begin(), ::toupper);
				Enable_GPU_Vaddr_Dump = (val.compare("FALSE") == 0 ? false : true);
			} else if (strcmp(param->name(), "L1Cache_Size") == 0) {
				std::string val = param->value();
				L1Cache_Size = std::stoi(val);
			} else if (strcmp(param->name(), "L1Cache_Assoc") == 0) {
				std::string val = param->value();
				L1Cache_Assoc = std::stoi(val);
			} else if (strcmp(param->name(), "L1Cache_Line_Size") == 0) {
				std::string val = param->value();
				L1Cache_Line_Size = std::stoi(val);
			} else if (strcmp(param->name(), "L1Cache_Banks") == 0) {
				std::string val = param->value();
				L1Cache_Banks = std::stoi(val);
			} else if (strcmp(param->name(), "L2Cache_Size") == 0) {
				std::string val = param->value();
				L2Cache_Size = std::stoi(val);
			} else if (strcmp(param->name(), "L2Cache_Assoc") == 0) {
				std::string val = param->value();
				L2Cache_Assoc = std::stoi(val);
			} else if (strcmp(param->name(), "L2Cache_Line_Size") == 0) {
				std::string val = param->value();
				L2Cache_Line_Size = std::stoi(val);
			} else if (strcmp(param->name(), "L2Cache_Banks") == 0) {
				std::string val = param->value();
				L2Cache_Banks = std::stoi(val);
			} else if (strcmp(param->name(), "Sharing_Size") == 0) {
				std::string val = param->value();
				Sharing_Size = std::stoi(val);
			} else if (strcmp(param->name(), "Sharing_Stride") == 0) {
				std::string val = param->value();
				Sharing_Stride = std::stoi(val);
			}
		}
	}
	catch (const std::exception& e)
	{
		std::cout << "Error Content: " << e.what() << std::endl;
		PRINT_ERROR("Error in GPU_Parameter_Set!")
	}
}
