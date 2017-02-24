# Algorithm information

import re, sys, numpy

from xml.dom import minidom

# implemented priors
re_prior_const=re.compile('constant')
re_prior_uni=re.compile('uniform')
re_prior_normal=re.compile('normal')
re_prior_logn=re.compile('lognormal') 

# True/False
re_true=re.compile('True')
re_none=re.compile('None')

def parse_required_single_value( node, tagname, message, cast ):
    try:
        data = node.getElementsByTagName(tagname)[0].firstChild.data
    except:
        print message
        sys.exit()

    ret = 0
    try:
        ret = cast( data )
    except:
        print message
        sys.exit()

    return(ret)

def parse_required_vector_value( node, tagname, message, cast ):
    try:
        data = node.getElementsByTagName(tagname)[0].firstChild.data
    except:
        print message
        sys.exit()

    tmp = str( data ).split()
    ret = []
    try:
        ret = [ cast(i) for i in tmp ]
    except:
        print message
        sys.exit()

    if len(ret) == 0:
        print message
        sys.exit()

    return(ret)

def process_prior( tmp ):
    prior_tmp = [0,0,0]

    if re_prior_const.match( tmp[0] ):
        prior_tmp[0] = 0
        try:
            prior_tmp[1] = float( tmp[1] )
        except:
            print "\nValue of the prior for model ", self.name[self.nmodels-1], "has the wrong format:", tmp[1]
            sys.exit()
                                
    elif re_prior_normal.match( tmp[0] ):
        prior_tmp[0] = 1
        try:
            prior_tmp[1] = float( tmp[1] )
            prior_tmp[2] = float( tmp[2] )
        except:
            print "\nValue of the prior for model ", self.name[self.nmodels-1], "has the wrong format:", tmp[1]
            sys.exit()

    elif re_prior_uni.match( tmp[0] ):
        prior_tmp[0] = 2
        try:
            prior_tmp[1] = float( tmp[1] )
            prior_tmp[2] = float( tmp[2] )
        except:
            print "\nValue of the prior for model ", self.name[self.nmodels-1], "has the wrong format:", tmp[1]
            sys.exit()
                                
    elif re_prior_logn.match( tmp[0] ):
        prior_tmp[0] = 3
        try:
            prior_tmp[1] = float( tmp[1] )
            prior_tmp[2] = float( tmp[2] )
        except:
            print "\nValue of the prior for model ", self.name[self.nmodels-1], "has the wrong format:", tmp[1]
            sys.exit()
    else:
        print "\nSupplied parameter prior ", tmp[0], " unsupported"
        sys.exit()

    return prior_tmp

def parse_fitting_information( node ):
    fitref = node.getElementsByTagName('fit')[0]
    tmp = str( fitref.firstChild.data ).split()
    ret = []

    if len(tmp) == 1 and re_none.match( tmp[0] ):
        return None
    else:
        for i in tmp:
            # replace species with samplePoints            
            ttmp = re.sub('species','samplePoints', i )

            # find all instances of samplePoints[ ] and extract the list of species numbers
            sp_strs = re.findall("samplePoints([0-9]+)",ttmp)
            sp_nums = [int(j) for j in sp_strs]
            sp_nums.sort()
            sp_nums.reverse()

            # loop over the species numbers and replace
            for n in sp_nums:
                ttmp = re.sub('ts'+str(n),'ts[:,'+str(n-1)+']',ttmp)

            ret.append( ttmp )

        return( ret )

class algorithm_info:
    """
    A class to parse the user-provided input file and return all information required to run the abc-SMC algorithm.
    
    """ 
    
    def __init__(self, filename, mode):
        xmldoc = minidom.parse(filename)
        self.mode = mode
        ### mode is 0  inference, 1 simulate, 2 design

        self.modelnumber = 0
        self.particles = 0
        self.beta = 0
        self.dt = 0
        self.times = []
        self.ntimes = 0
        
        
        self.nmodels = 0
        self.nparameters = []
        self.nspecies = []
        self.name = []
        self.source = []
        self.type = []
        self.prior = []
        self.x0prior = []
        self.fit = []


        ##################################################
        ## Required arguments

        ### get number of models
        self.modelnumber = parse_required_single_value( xmldoc, "modelnumber", "Please provide an integer value for <modelnumber>", int )

        ### get number of samples
        self.particles = parse_required_single_value( xmldoc, "particles", "Please provide an integer value for <particles>", int )


        ### get dt
        self.dt = parse_required_single_value( xmldoc, "dt", "Please provide an float value for <dt>", float )

  
        ### get data attributes
        dataref = xmldoc.getElementsByTagName('data')[0]
        # times
        self.times = parse_required_vector_value( dataref, "times", "Please provide a whitespace separated list of values for <data><times>" , float )
        self.ntimes = len(self.times)

           
        ### get model attributes
        modelref = xmldoc.getElementsByTagName('models')[0]
        for m in modelref.childNodes:
            if m.nodeType == m.ELEMENT_NODE:
                self.nmodels += 1
                self.prior.append([])
                self.x0prior.append([])

                self.name.append( str(m.getElementsByTagName('name')[0].firstChild.data).strip() )
                self.source.append( str(m.getElementsByTagName('source')[0].firstChild.data).strip() )
                self.type.append( str(m.getElementsByTagName('type')[0].firstChild.data).strip() )
                
                self.fit.append( parse_fitting_information( m )  )



                #initref = m.getElementsByTagName('initialvalues')[0]
                #tmp = str( initref.firstChild.data ).split()
                #self.init.append( [ float(i) for i in tmp ] )
                #self.nspecies.append( len( self.init[self.nmodels-1] ) )

                nparameter = 0
                paramref = m.getElementsByTagName('parameters')[0]
                for p in paramref.childNodes:
                    if p.nodeType == p.ELEMENT_NODE:
                        nparameter += 1
                        prior_tmp = [0,0,0]
                        tmp = str( p.firstChild.data ).split()
                        self.prior[self.nmodels-1].append( process_prior( tmp ) )

                ninit = 0
                initref = m.getElementsByTagName('initial')[0]
                for inn in initref.childNodes:
                    if inn.nodeType == inn.ELEMENT_NODE:
                        ninit += 1
                        prior_tmp = [0,0,0]
                        tmp = str( inn.firstChild.data ).split()
                        self.x0prior[self.nmodels-1].append( process_prior( tmp ) )

                if nparameter == 0:
                    print "\nNo parameters specified in model ", self.name[self.nmodels-1]
                    sys.exit()
                if ninit == 0:
                    print "\nNo initial conditions specified in model ", self.name[self.nmodels-1]
                    sys.exit()
                self.nparameters.append( nparameter )
                self.nspecies.append( ninit )
                
        if self.nmodels == 0:
            print "\nNo models specified"
            sys.exit()


          

    def print_info(self):
        print "\nALGORITHM INFO"
        print "modelnumber:", self.modelnumber
        print "samples:", self.particles
        print "dt:", self.dt

        
        print "\ttimes:", self.times

        
        print "MODELS:", self.nmodels
        for i in range(self.nmodels):
            print "\t", "npar:", self.nparameters[i]
            print "\t", "nspecies:", self.nspecies[i]
            print "\t", "name:", self.name[i]
            print "\t", "source:", self.source[i]
            print "\t", "type:", self.type[i]
            print "\t", "fit:", self.fit[i]
            print "\t", "init:", self.x0prior[i]
            print "\t", "prior:", self.prior[i]
            print "\n"

