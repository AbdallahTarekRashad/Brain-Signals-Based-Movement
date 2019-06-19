class Data:
    def __init__(self, path='MNE', subject=None, run=None, class_id={'right': 2, 'left': 3}):
        if path == "MNE":
            self.path = 'MNE'
            if subject is None:
                raise ValueError("subject should be of type int if path is MNE (%s)."
                                 % type(subject))
            self.subject = subject

            if run is None:
                raise ValueError("run should be of type list if path is MNE (%s)."
                                 % type(subject))
            self.run = run
            self.class_id = class_id
        else:
            self.path = "data/"+path
            self.class_id = class_id

        self.raw            = None
        self.data           = None
        self.epochs         = None
        self.labels         = None
        self.t_min          = None
        self.t_max          = None
        self.l_freq         = None
        self.h_freq         = None
        self.pick_channels  = None
        self.events         = None
        self.montage        = None
        self.close_eye      = None
        self.open_eye       = None
        self.raw_data()

    def change_subject(self, subject):
        '''
        To change the subject of an object without creating another object and give them the same configuration
        
        In developing for our expriment  
          
        '''
        if self.path == 'MNE':
          self.subject = subject
        else:
          self.path = "data/"+subject
          
        self.raw_data()
        self.get_epochs(self.t_min, self.t_max)
        
        
        if self.l_freq is not None and self.h_freq is not None:
            self.filter(self.l_freq, self.h_freq)
        if self.pick_channels is not None:
            self.pick_channel(self.pick_channels)
        self.get_data()
        if self.open_eye is not None or self.close_eye is not None :
            self.baseline_subtraction()

    def raw_data(self):
        '''
        read raw data and set montage for it
        
        In developing for our expriment
        '''
        if self.path == 'MNE':
            channels = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2',
                        'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz',
                        'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2',
                        'F4', 'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7',
                        'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4',
                        'Po8', 'O1', 'Oz', 'O2', 'Iz']
            files_names = eegbci.load_data(self.subject, self.run)
            raws_data = [read_raw_edf(f, preload=True,stim_channel="auto") for f in files_names]
            self.raw = concatenate_raws(raws_data)
            self.raw.rename_channels(lambda x: x.strip('.'))
            self.montage = mne.channels.read_montage(kind='standard_1020', ch_names=channels,  unit='m', transform=False)
            self.raw.set_montage(self.montage, set_dig=True)
        else:
            files=os.listdir(self.path)
            raws_data = [read_raw_edf(self.path+"/"+f, preload=True,stim_channel="auto") for f in files]
            self.raw = concatenate_raws(raws_data)
        
        
    def raw_data_all(self,sub):
        '''
        read raw data and set montage for it
        
        In developing for our expriment
        '''
        files_names = []
        if self.path == 'MNE':
            channels = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2',
                        'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz',
                        'Fp2', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2',
                        'F4', 'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7',
                        'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4',
                        'Po8', 'O1', 'Oz', 'O2', 'Iz']
            for i in sub:
              files_names.extend(eegbci.load_data(i, self.run))
            raws_data = [read_raw_edf(f, preload=True) for f in files_names]
            self.raw = concatenate_raws(raws_data)
            self.raw.rename_channels(lambda x: x.strip('.'))
            self.montage = mne.channels.read_montage(kind='standard_1020', ch_names=channels,  unit='m', transform=False)
            self.raw.set_montage(self.montage, set_dig=True)
        else:
          files_data=list()
          for i in names:
            sub_files_names=os.listdir('data/' + i)
            for sub_file in sub_files_names:
              sub_file = 'data/' + i + '/' + sub_file
              files_data.append(sub_file)
            files_names.extend(files_data)
            raws_data = [read_raw_edf(f, preload=True,stim_channel="auto") for f in files_names]
            self.raw = concatenate_raws(raws_data)
    
    def baseline_subtraction(self):
        '''
        To subtract baseline from tril
        '''
        events = mne.find_events(self.raw, verbose=True)
        picks = mne.pick_types(self.raw.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude='bads')
        self.close_eye = mne.Epochs(self.raw, events, {'close':9}, tmin=-7, tmax=-3, proj=True, picks=picks,
                                baseline=None, preload=True).get_data()[0]
        self.open_eye = mne.Epochs(self.raw, events, {'open':6}, tmin=3, tmax=7, proj=True, picks=picks,
                                baseline=None, preload=True).get_data()[0]
        
        self.data = self.data - self.open_eye
    
    def get_epochs(self, t_min= 0, t_max=4):
        '''
        To cut off data to trials by the time interval
        '''
        self.t_min = t_min
        self.t_max = t_max
        self.events = mne.find_events(self.raw, verbose=True)
        picks = mne.pick_types(self.raw.info, meg=False, eeg=True, stim=False, eog=False,
                               exclude='bads')
        self.epochs = mne.Epochs(self.raw, self.events, self.class_id, tmin=t_min, tmax=t_max, proj=True, picks=picks,
                                baseline=None, preload=True)

    def filter(self, l_freq, h_freq):
        '''
        Filtering data by bandpass filter
        '''
        self.l_freq = l_freq
        self.h_freq = h_freq
        if self.epochs is None:
            raise ValueError("you should run get_epochs before filter")
        self.epochs.filter(l_freq, h_freq, fir_design='firwin')
        

    def pick_channel(self, channels):
        '''
        pick list of channels from your data 
        '''
        if self.epochs is None:
            raise ValueError("you should run get_epochs before pick_channel")
        self.pick_channels = channels
        self.epochs.pick_channels(self.pick_channels)

    def get_data(self):
        '''
        after cat off your data this function give you data and labels of subject
        '''
        if self.epochs is None:
            raise ValueError("you should run get_epochs before get_data")
        self.data = self.epochs.get_data()
        self.labels = self.epochs.events[:, -1] - self.epochs.events[:, -1].min()
        return self.data, self.labels