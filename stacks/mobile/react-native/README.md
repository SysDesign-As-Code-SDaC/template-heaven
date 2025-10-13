# React Native Template

A production-ready React Native application template with TypeScript, navigation, state management, and testing.

## 🚀 Features

- **React Native 0.72+** with latest features
- **TypeScript** for type safety
- **React Navigation** for navigation
- **Redux Toolkit** for state management
- **React Query** for data fetching
- **NativeBase** for UI components
- **Jest & Detox** for testing
- **Flipper** integration for debugging
- **Fastlane** for deployment automation
- **CodePush** for over-the-air updates

## 📋 Prerequisites

- Node.js 18+
- React Native CLI
- Android Studio (for Android development)
- Xcode (for iOS development, macOS only)
- CocoaPods (for iOS dependencies)

## 🛠️ Quick Start

### 1. Create New Project

```bash
npx react-native@latest init MyApp --template react-native-template-typescript
cd MyApp
```

### 2. Install Dependencies

```bash
npm install
# or
yarn install

# For iOS
cd ios && pod install && cd ..
```

### 3. Start Development Server

```bash
# Start Metro bundler
npm run start

# Run on Android
npm run android

# Run on iOS
npm run ios
```

## 📁 Project Structure

```
├── src/
│   ├── components/         # Reusable UI components
│   ├── screens/           # Screen components
│   ├── navigation/        # Navigation configuration
│   ├── store/            # Redux store and slices
│   ├── services/         # API services
│   ├── hooks/            # Custom React hooks
│   ├── utils/            # Utility functions
│   ├── types/            # TypeScript type definitions
│   ├── constants/        # App constants
│   └── assets/           # Images, fonts, etc.
├── android/              # Android-specific code
├── ios/                  # iOS-specific code
├── __tests__/            # Test files
└── fastlane/             # Deployment automation
```

## 🔧 Available Scripts

```bash
# Development
npm run start             # Start Metro bundler
npm run android           # Run on Android
npm run ios               # Run on iOS
npm run web               # Run on web (if configured)

# Testing
npm run test              # Run unit tests
npm run test:watch        # Run tests in watch mode
npm run test:e2e          # Run E2E tests with Detox
npm run test:coverage     # Run tests with coverage

# Building
npm run build:android     # Build Android APK
npm run build:ios         # Build iOS app
npm run build:release     # Build release versions

# Deployment
npm run deploy:android    # Deploy to Google Play
npm run deploy:ios        # Deploy to App Store
npm run codepush:android  # Push update via CodePush
npm run codepush:ios      # Push update via CodePush
```

## 🧭 Navigation Setup

```typescript
// src/navigation/AppNavigator.tsx
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import HomeScreen from '../screens/HomeScreen';
import ProfileScreen from '../screens/ProfileScreen';
import SettingsScreen from '../screens/SettingsScreen';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

function TabNavigator() {
  return (
    <Tab.Navigator>
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

export default function AppNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Main" component={TabNavigator} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

## 🗄️ State Management with Redux Toolkit

```typescript
// src/store/slices/userSlice.ts
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { User } from '../types';

interface UserState {
  user: User | null;
  loading: boolean;
  error: string | null;
}

const initialState: UserState = {
  user: null,
  loading: false,
  error: null,
};

export const fetchUser = createAsyncThunk(
  'user/fetchUser',
  async (userId: string) => {
    const response = await fetch(`/api/users/${userId}`);
    return response.json();
  }
);

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    clearUser: (state) => {
      state.user = null;
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUser.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchUser.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
      })
      .addCase(fetchUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch user';
      });
  },
});

export const { clearUser } = userSlice.actions;
export default userSlice.reducer;
```

## 🌐 API Integration with React Query

```typescript
// src/services/api.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export const useUsers = () => {
  return useQuery({
    queryKey: ['users'],
    queryFn: async () => {
      const response = await fetch('/api/users');
      return response.json();
    },
  });
};

export const useCreateUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (userData: CreateUserData) => {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData),
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
};
```

## 🎨 UI Components with NativeBase

```typescript
// src/components/UserCard.tsx
import React from 'react';
import { Box, Text, Avatar, Button } from 'native-base';
import { User } from '../types';

interface UserCardProps {
  user: User;
  onPress: () => void;
}

export const UserCard: React.FC<UserCardProps> = ({ user, onPress }) => {
  return (
    <Box
      bg="white"
      shadow={2}
      rounded="lg"
      p={4}
      mb={3}
    >
      <Box flexDirection="row" alignItems="center">
        <Avatar
          source={{ uri: user.avatar }}
          size="md"
          mr={3}
        />
        <Box flex={1}>
          <Text fontSize="lg" fontWeight="bold">
            {user.name}
          </Text>
          <Text color="gray.500">
            {user.email}
          </Text>
        </Box>
        <Button
          size="sm"
          variant="outline"
          onPress={onPress}
        >
          View
        </Button>
      </Box>
    </Box>
  );
};
```

## 🧪 Testing

### Unit Testing with Jest

```typescript
// __tests__/components/UserCard.test.tsx
import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { UserCard } from '../src/components/UserCard';

const mockUser = {
  id: '1',
  name: 'John Doe',
  email: 'john@example.com',
  avatar: 'https://example.com/avatar.jpg',
};

describe('UserCard', () => {
  it('renders user information correctly', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <UserCard user={mockUser} onPress={onPress} />
    );
    
    expect(getByText('John Doe')).toBeTruthy();
    expect(getByText('john@example.com')).toBeTruthy();
  });

  it('calls onPress when button is pressed', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <UserCard user={mockUser} onPress={onPress} />
    );
    
    fireEvent.press(getByText('View'));
    expect(onPress).toHaveBeenCalled();
  });
});
```

### E2E Testing with Detox

```typescript
// e2e/firstTest.e2e.ts
import { device, expect, element, by } from 'detox';

describe('Example', () => {
  beforeAll(async () => {
    await device.launchApp();
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it('should show welcome screen', async () => {
    await expect(element(by.id('welcome'))).toBeVisible();
  });

  it('should navigate to profile screen', async () => {
    await element(by.id('profile-tab')).tap();
    await expect(element(by.id('profile-screen'))).toBeVisible();
  });
});
```

## 🚀 Deployment with Fastlane

```ruby
# fastlane/Fastfile
default_platform(:android)

platform :android do
  desc "Deploy to Google Play Store"
  lane :deploy do
    gradle(
      task: "bundle",
      build_type: "Release"
    )
    
    upload_to_play_store(
      track: "internal",
      aab: "android/app/build/outputs/bundle/release/app-release.aab"
    )
  end
end

platform :ios do
  desc "Deploy to App Store"
  lane :deploy do
    build_app(
      scheme: "MyApp",
      export_method: "app-store"
    )
    
    upload_to_app_store(
      skip_metadata: true,
      skip_screenshots: true
    )
  end
end
```

## 📱 CodePush Integration

```typescript
// src/services/codePush.ts
import codePush from 'react-native-code-push';

export const checkForUpdate = async () => {
  try {
    const update = await codePush.checkForUpdate();
    if (update) {
      const result = await codePush.sync({
        updateDialog: {
          title: 'Update Available',
          optionalInstallButtonLabel: 'Install',
          optionalIgnoreButtonLabel: 'Later',
        },
        installMode: codePush.InstallMode.IMMEDIATE,
      });
      return result;
    }
  } catch (error) {
    console.error('CodePush update failed:', error);
  }
};
```

## 📚 Learning Resources

- [React Native Documentation](https://reactnative.dev/)
- [React Navigation Documentation](https://reactnavigation.org/)
- [Redux Toolkit Documentation](https://redux-toolkit.js.org/)
- [NativeBase Documentation](https://nativebase.io/)
- [Detox Documentation](https://wix.github.io/Detox/)

## 🔗 Upstream Source

- **Repository**: [react-native-community/react-native-template-typescript](https://github.com/react-native-community/react-native-template-typescript)
- **Documentation**: [reactnative.dev](https://reactnative.dev/)
- **License**: MIT
