# Hardhat DApp Template

A production-ready decentralized application (DApp) template using Hardhat for smart contract development, featuring modern Web3 tooling and best practices for 2025.

## 🚀 Features

- **Hardhat** - Ethereum development environment
- **TypeScript** - Type-safe smart contract development
- **OpenZeppelin** - Secure smart contract libraries
- **Ethers.js** - Ethereum library for frontend integration
- **Wagmi** - React hooks for Ethereum
- **RainbowKit** - Beautiful wallet connection UI
- **Next.js** - Modern React framework for frontend
- **Tailwind CSS** - Utility-first CSS framework
- **Testing Suite** - Comprehensive smart contract testing
- **Gas Optimization** - Optimized contract deployment
- **Multi-Network Support** - Ethereum, Polygon, Arbitrum, etc.

## 📋 Prerequisites

- Node.js 18+
- Git
- MetaMask or compatible wallet
- Alchemy/Infura API key (for mainnet/testnet)

## 🛠️ Quick Start

### 1. Create New DApp

```bash
# Clone the template
git clone <this-repo> my-dapp
cd my-dapp

# Or use Hardhat directly
npx hardhat init
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Environment Setup

```bash
cp .env.example .env
```

Configure your environment variables:

```env
# Network Configuration
MAINNET_RPC_URL=https://eth-mainnet.alchemyapi.io/v2/YOUR_API_KEY
GOERLI_RPC_URL=https://eth-goerli.alchemyapi.io/v2/YOUR_API_KEY
POLYGON_RPC_URL=https://polygon-mainnet.alchemyapi.io/v2/YOUR_API_KEY

# Private Keys (for deployment)
PRIVATE_KEY=your_private_key_here

# Etherscan API Key (for contract verification)
ETHERSCAN_API_KEY=your_etherscan_api_key

# Alchemy API Key
ALCHEMY_API_KEY=your_alchemy_api_key

# Pinata API (for IPFS)
PINATA_API_KEY=your_pinata_api_key
PINATA_SECRET_KEY=your_pinata_secret_key
```

### 4. Compile Contracts

```bash
npx hardhat compile
```

### 5. Run Tests

```bash
npx hardhat test
```

### 6. Deploy Contracts

```bash
# Deploy to local network
npx hardhat run scripts/deploy.ts --network localhost

# Deploy to testnet
npx hardhat run scripts/deploy.ts --network goerli

# Deploy to mainnet
npx hardhat run scripts/deploy.ts --network mainnet
```

## 📁 Project Structure

```
├── contracts/                 # Smart contracts
│   ├── Token.sol             # ERC-20 token contract
│   ├── NFT.sol               # ERC-721 NFT contract
│   ├── Marketplace.sol       # NFT marketplace contract
│   └── interfaces/           # Contract interfaces
├── scripts/                  # Deployment and utility scripts
│   ├── deploy.ts             # Contract deployment script
│   ├── verify.ts             # Contract verification script
│   └── mint.ts               # NFT minting script
├── test/                     # Smart contract tests
│   ├── Token.test.ts
│   ├── NFT.test.ts
│   └── Marketplace.test.ts
├── frontend/                 # Next.js frontend application
│   ├── components/           # React components
│   ├── hooks/               # Custom React hooks
│   ├── pages/               # Next.js pages
│   └── styles/              # CSS styles
├── hardhat.config.ts         # Hardhat configuration
└── package.json              # Dependencies and scripts
```

## 🔧 Available Scripts

```bash
# Smart Contract Development
npx hardhat compile           # Compile contracts
npx hardhat test              # Run tests
npx hardhat test --grep "Token" # Run specific tests
npx hardhat coverage          # Generate test coverage
npx hardhat clean             # Clean build artifacts

# Deployment
npx hardhat run scripts/deploy.ts --network localhost
npx hardhat run scripts/deploy.ts --network goerli
npx hardhat run scripts/deploy.ts --network mainnet

# Verification
npx hardhat verify --network goerli <CONTRACT_ADDRESS> <CONSTRUCTOR_ARGS>

# Frontend Development
cd frontend
npm run dev                   # Start development server
npm run build                 # Build for production
npm run start                 # Start production server
npm run lint                  # Run ESLint
```

## 📝 Smart Contract Example

```solidity
// contracts/Token.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract MyToken is ERC20, Ownable, Pausable {
    uint256 public constant MAX_SUPPLY = 1000000 * 10**18;
    uint256 public constant INITIAL_SUPPLY = 100000 * 10**18;
    
    mapping(address => bool) public minters;
    
    event MinterAdded(address indexed minter);
    event MinterRemoved(address indexed minter);
    
    constructor() ERC20("MyToken", "MTK") {
        _mint(msg.sender, INITIAL_SUPPLY);
    }
    
    modifier onlyMinter() {
        require(minters[msg.sender] || msg.sender == owner(), "Not a minter");
        _;
    }
    
    function addMinter(address _minter) external onlyOwner {
        minters[_minter] = true;
        emit MinterAdded(_minter);
    }
    
    function removeMinter(address _minter) external onlyOwner {
        minters[_minter] = false;
        emit MinterRemoved(_minter);
    }
    
    function mint(address to, uint256 amount) external onlyMinter whenNotPaused {
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
    }
    
    function pause() external onlyOwner {
        _pause();
    }
    
    function unpause() external onlyOwner {
        _unpause();
    }
    
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }
}
```

## 🧪 Testing Smart Contracts

```typescript
// test/Token.test.ts
import { expect } from "chai";
import { ethers } from "hardhat";
import { MyToken } from "../typechain-types";

describe("MyToken", function () {
  let token: MyToken;
  let owner: any;
  let addr1: any;
  let addr2: any;

  beforeEach(async function () {
    [owner, addr1, addr2] = await ethers.getSigners();
    
    const Token = await ethers.getContractFactory("MyToken");
    token = await Token.deploy();
    await token.deployed();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await token.owner()).to.equal(owner.address);
    });

    it("Should assign the total supply of tokens to the owner", async function () {
      const ownerBalance = await token.balanceOf(owner.address);
      expect(await token.totalSupply()).to.equal(ownerBalance);
    });

    it("Should have correct name and symbol", async function () {
      expect(await token.name()).to.equal("MyToken");
      expect(await token.symbol()).to.equal("MTK");
    });
  });

  describe("Minting", function () {
    it("Should allow owner to mint tokens", async function () {
      const mintAmount = ethers.utils.parseEther("1000");
      await token.mint(addr1.address, mintAmount);
      
      expect(await token.balanceOf(addr1.address)).to.equal(mintAmount);
    });

    it("Should not allow non-minters to mint", async function () {
      const mintAmount = ethers.utils.parseEther("1000");
      
      await expect(
        token.connect(addr1).mint(addr2.address, mintAmount)
      ).to.be.revertedWith("Not a minter");
    });

    it("Should not exceed max supply", async function () {
      const maxSupply = await token.MAX_SUPPLY();
      const currentSupply = await token.totalSupply();
      const excessAmount = maxSupply.sub(currentSupply).add(1);
      
      await expect(
        token.mint(addr1.address, excessAmount)
      ).to.be.revertedWith("Exceeds max supply");
    });
  });

  describe("Pausing", function () {
    it("Should allow owner to pause", async function () {
      await token.pause();
      expect(await token.paused()).to.be.true;
    });

    it("Should prevent transfers when paused", async function () {
      await token.pause();
      const transferAmount = ethers.utils.parseEther("100");
      
      await expect(
        token.transfer(addr1.address, transferAmount)
      ).to.be.revertedWith("Pausable: paused");
    });
  });
});
```

## 🎨 Frontend Integration

```typescript
// frontend/hooks/useContract.ts
import { useContract, useProvider, useSigner } from 'wagmi';
import { ethers } from 'ethers';

export const useMyToken = () => {
  const { data: signer } = useSigner();
  const provider = useProvider();
  
  const contract = useContract({
    address: process.env.NEXT_PUBLIC_TOKEN_ADDRESS,
    abi: [
      "function name() view returns (string)",
      "function symbol() view returns (string)",
      "function totalSupply() view returns (uint256)",
      "function balanceOf(address) view returns (uint256)",
      "function mint(address to, uint256 amount)",
      "function addMinter(address minter)",
      "function removeMinter(address minter)",
      "function pause()",
      "function unpause()",
      "function paused() view returns (bool)",
      "event Transfer(address indexed from, address indexed to, uint256 value)",
      "event MinterAdded(address indexed minter)",
      "event MinterRemoved(address indexed minter)"
    ],
    signerOrProvider: signer || provider,
  });

  return contract;
};
```

```typescript
// frontend/components/TokenMinter.tsx
import { useState } from 'react';
import { useAccount, useMyToken } from '../hooks/useContract';
import { ethers } from 'ethers';

export const TokenMinter = () => {
  const { address } = useAccount();
  const token = useMyToken();
  const [mintAmount, setMintAmount] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleMint = async () => {
    if (!token || !address) return;
    
    setIsLoading(true);
    try {
      const amount = ethers.utils.parseEther(mintAmount);
      const tx = await token.mint(address, amount);
      await tx.wait();
      alert('Tokens minted successfully!');
    } catch (error) {
      console.error('Minting failed:', error);
      alert('Minting failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-4">Mint Tokens</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Amount to Mint
          </label>
          <input
            type="number"
            value={mintAmount}
            onChange={(e) => setMintAmount(e.target.value)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            placeholder="Enter amount"
          />
        </div>
        <button
          onClick={handleMint}
          disabled={isLoading || !mintAmount}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50"
        >
          {isLoading ? 'Minting...' : 'Mint Tokens'}
        </button>
      </div>
    </div>
  );
};
```

## 🚀 Deployment Script

```typescript
// scripts/deploy.ts
import { ethers } from "hardhat";

async function main() {
  const [deployer] = await ethers.getSigners();
  
  console.log("Deploying contracts with the account:", deployer.address);
  console.log("Account balance:", (await deployer.getBalance()).toString());

  // Deploy Token
  const Token = await ethers.getContractFactory("MyToken");
  const token = await Token.deploy();
  await token.deployed();
  
  console.log("Token deployed to:", token.address);

  // Deploy NFT
  const NFT = await ethers.getContractFactory("MyNFT");
  const nft = await NFT.deploy();
  await nft.deployed();
  
  console.log("NFT deployed to:", nft.address);

  // Deploy Marketplace
  const Marketplace = await ethers.getContractFactory("Marketplace");
  const marketplace = await Marketplace.deploy(nft.address);
  await marketplace.deployed();
  
  console.log("Marketplace deployed to:", marketplace.address);

  // Save contract addresses
  const fs = require("fs");
  const contracts = {
    token: token.address,
    nft: nft.address,
    marketplace: marketplace.address,
    network: await ethers.provider.getNetwork(),
  };
  
  fs.writeFileSync(
    "./frontend/contracts.json",
    JSON.stringify(contracts, null, 2)
  );
  
  console.log("Contract addresses saved to frontend/contracts.json");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
```

## 🔒 Security Best Practices

### Contract Security

```solidity
// Use OpenZeppelin contracts for security
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

// Implement proper access controls
contract SecureContract is ReentrancyGuard, Pausable, AccessControl {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }
    
    function secureFunction() external onlyRole(MINTER_ROLE) nonReentrant whenNotPaused {
        // Implementation
    }
}
```

### Frontend Security

```typescript
// Validate contract addresses
const validateAddress = (address: string): boolean => {
  return ethers.utils.isAddress(address);
};

// Use proper error handling
try {
  const tx = await contract.someFunction();
  await tx.wait();
} catch (error) {
  if (error.code === 'USER_REJECTED') {
    // User rejected the transaction
  } else if (error.code === 'INSUFFICIENT_FUNDS') {
    // Insufficient funds
  } else {
    // Other errors
  }
}
```

## 📚 Learning Resources

- [Hardhat Documentation](https://hardhat.org/docs)
- [OpenZeppelin Contracts](https://docs.openzeppelin.com/contracts/)
- [Ethers.js Documentation](https://docs.ethers.io/)
- [Wagmi Documentation](https://wagmi.sh/)
- [Solidity Documentation](https://docs.soliditylang.org/)

## 🔗 Upstream Source

- **Repository**: [Nomic Foundation Hardhat](https://github.com/NomicFoundation/hardhat)
- **Documentation**: [hardhat.org](https://hardhat.org/)
- **License**: MIT
