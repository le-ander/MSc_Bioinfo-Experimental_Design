import re
import sys

def stringSearch(orig_str,orig_param,replacement):
	
	not_an = re.compile(r"[^A-Za-z0-9]")
	not_space = re.compile(r"[-\s]")
	replaced_string = ""

	if orig_str[:len(orig_param)] == orig_param and (not_an.search(orig_str[len(orig_param)]) or not_space.search(orig_str[len(orig_param)])):
		replaced_string+=replacement
		start = len(orig_param)
	else:
		replaced_string+=orig_str[0]
		start = 1

	if orig_str[-len(orig_param):] == orig_param and (not_an.search(orig_str[-len(orig_param)-1]) or not_space.search(orig_str[-len(orig_param)-1])):
		replaced_string_end=replacement
		end = len(orig_str)-len(orig_param)
	else:
		replaced_string_end=orig_str[-1]
		end = -1

	i = start

	while i < start+len(orig_str[start:end]):
	#for i, param in enumerate(orig_str[start:end]):
		if orig_str[i:i+len(orig_param)] == orig_param and (not_an.search(orig_str[i+len(orig_param)]) or not_space.search(orig_str[i+len(orig_param)])) and (not_an.search(orig_str[i-1]) or not_space.search(orig_str[i-1])):
			replaced_string+=replacement
			i+=len(orig_param)
		else:
			replaced_string+=orig_str[i]
			i+=1

	replaced_string += replaced_string_end

	return replaced_string
	

# --- Main

string_to_be_changed   ="alpha0 + alpha / (1 + pow(p3, h)) + alpha"
string_to_search_for   = "alpha0"
string_to_replace_with = "10 * alpha0"

string_result = stringSearch(string_to_be_changed,string_to_search_for,string_to_replace_with)

print " Here is the final substituted string:"
print string_result
